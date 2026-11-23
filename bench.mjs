import { spawn, spawnSync } from "node:child_process";
import { appendFileSync, readFileSync } from "node:fs";
import { createServer } from "node:net";
import { dirname } from "node:path";

const flag = (name) => process.argv.find((arg) => arg.startsWith(`--${name}=`))?.slice(name.length + 3);
const CONFIGS = JSON.parse(readFileSync(flag("configs-file") ?? "configs.json", "utf8"));
const MODEL = flag("model");
const SERVERS = { cuda: flag("cuda"), vulkan: flag("vulkan"), cpu: flag("vulkan") };
const QUICK = Boolean(flag("quick"));
const pause = (ms) => new Promise((done) => setTimeout(done, ms));
const ASKS = ["Hi! In one sentence, who are you?", "Write a short friendly email telling my team the launch moved to Friday.", "Explain what a hash map is to a new programmer in about 120 words."];

if (!MODEL || !Object.values(SERVERS).some(Boolean)) {
	console.log("node bench.mjs --model=path/to/model.gguf --cuda=path/to/cuda/llama-server.exe [--vulkan=path/to/vulkan/llama-server.exe] --configs=before,final [--quick=1]");
	process.exit(1);
}

const vram = () => Number(spawnSync("nvidia-smi", ["--query-gpu=memory.used", "--format=csv,noheader,nounits"], { encoding: "utf8" }).stdout?.trim()) || 0;
const ram = (pid) => {
	const script = process.platform === "win32" ? ["powershell.exe", ["-NoProfile", "-Command", `(Get-Process -Id ${pid}).WorkingSet64`]] : ["ps", ["-o", "rss=", "-p", String(pid)]];
	const value = Number(spawnSync(script[0], script[1], { encoding: "utf8" }).stdout.trim());
	return Math.round(process.platform === "win32" ? value / 2 ** 20 : value / 1024);
};
const freePort = () => new Promise((done) => { const server = createServer().listen(0, "127.0.0.1", () => { const { port } = server.address(); server.close(() => done(port)); }); });

async function ask(port, content, think) {
	const at = performance.now();
	const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
		body: JSON.stringify({ chat_template_kwargs: { enable_thinking: think }, max_tokens: 700, messages: [{ content, role: "user" }], repeat_penalty: 1.1, stream: true, temperature: 0.1, top_k: 50 }),
		headers: { "content-type": "application/json" },
		method: "POST",
	});
	const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
	let answer = 0;
	let timings = {};
	let carry = "";
	for (;;) {
		const { done, value } = await reader.read();
		if (done) break;
		carry += value;
		const lines = carry.split("\n");
		carry = lines.pop();
		for (const line of lines) {
			if (!line.startsWith("data:")) continue;
			const data = line.slice(5).trim();
			if (!data || data === "[DONE]") continue;
			const event = JSON.parse(data);
			if (event.choices?.[0]?.delta?.content && !answer) answer = performance.now() - at;
			if (event.timings) timings = event.timings;
		}
	}
	return { answerMs: Math.round(answer), promptTps: Math.round(timings.prompt_per_second ?? 0), think, totalMs: Math.round(performance.now() - at), tps: Math.round(timings.predicted_per_second ?? 0) };
}

for (const name of (flag("configs") ?? "before,final").split(",")) {
	const config = CONFIGS[name];
	const exe = SERVERS[config.backend];
	if (!exe) {
		console.log(`${name}: no --${config.backend} server given`);
		continue;
	}
	const port = await freePort();
	const idle = vram();
	const started = performance.now();
	const child = spawn(exe, ["-m", MODEL, "--host", "127.0.0.1", "--port", String(port), ...config.args], { cwd: dirname(exe), env: { ...process.env, ...config.env }, stdio: "ignore" });
	let ready = false;
	while (!ready && performance.now() - started < 180_000 && child.exitCode === null) {
		ready = await fetch(`http://127.0.0.1:${port}/health`).then((r) => r.json()).then((body) => body.status === "ok").catch(() => false);
		if (!ready) await pause(100);
	}
	if (!ready) {
		console.log(`${name}: did not start`);
		child.kill();
		continue;
	}
	const loadMs = Math.round(performance.now() - started);
	await ask(port, "Hi", false);
	const rows = [];
	for (const think of QUICK || config.quick ? [false] : [false, true]) for (const content of ASKS) rows.push(await ask(port, content, think));
	const summary = { config: name, loadMs, ramMb: ram(child.pid), vramMb: vram() - idle };
	child.kill();
	await pause(1_500);
	const mean = (list, key) => (list.length ? Math.round(list.reduce((sum, row) => sum + row[key], 0) / list.length) : 0);
	Object.assign(summary, { answerMs: mean(rows.filter((row) => !row.think), "answerMs"), promptTps: mean(rows, "promptTps"), thinkAnswerMs: mean(rows.filter((row) => row.think), "answerMs"), tps: mean(rows, "tps") });
	console.log(`${name}: ${summary.tps} tok/s, prompt ${summary.promptTps} tok/s, first answer word ${summary.answerMs} ms${summary.thinkAnswerMs ? ` (${summary.thinkAnswerMs} ms with thinking)` : ""}, load ${loadMs} ms, VRAM +${summary.vramMb} MB, RAM ${summary.ramMb} MB`);
	appendFileSync(flag("out") ?? "results.jsonl", `${JSON.stringify({ at: new Date().toISOString(), model: MODEL.split(/[\/]/).pop(), ...summary, args: config.args.join(" ") })}\n`);
}
