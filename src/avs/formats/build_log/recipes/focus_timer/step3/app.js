// Focus timer: measures elapsed time against a start timestamp, so the
// display stays correct even when the browser throttles timers.
const FOCUS_SECONDS = 25 * 60;
const speed = Number(new URLSearchParams(location.search).get("speed")) || 1; // demo/test speed-up

const timeEl = document.getElementById("time");
const startBtn = document.getElementById("start");
const resetBtn = document.getElementById("reset");
const ring = document.querySelector("circle.progress");
const sessionsEl = document.getElementById("sessions");
const CIRCUMFERENCE = 2 * Math.PI * 100;
let sessions = Number(localStorage.getItem("focus.sessions") || 0);

let remaining = FOCUS_SECONDS;
let startedAt = null;   // ms timestamp when the current run started
let atStart = remaining; // seconds remaining when the current run started
let ticker = null;

function format(seconds) {
  const s = Math.max(0, Math.ceil(seconds));
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

function render() {
  timeEl.textContent = format(remaining);
  startBtn.textContent = ticker ? "Pause" : "Start";
  ring.style.strokeDashoffset = CIRCUMFERENCE * (1 - remaining / FOCUS_SECONDS);
  sessionsEl.textContent = sessions;
}

function complete() {
  sessions += 1;
  localStorage.setItem("focus.sessions", sessions);
}

function tick() {
  const elapsed = ((Date.now() - startedAt) / 1000) * speed;
  remaining = atStart - elapsed;
  if (remaining <= 0) {
    remaining = 0;
    pause();
    complete();
  }
  render();
}

function start() {
  startedAt = Date.now();
  atStart = remaining;
  ticker = setInterval(tick, 250);
  render();
}

function pause() {
  clearInterval(ticker);
  ticker = null;
  render();
}

function reset() {
  pause();
  remaining = FOCUS_SECONDS;
  render();
}

startBtn.addEventListener("click", () => (ticker ? pause() : start()));
resetBtn.addEventListener("click", reset);
document.addEventListener("keydown", (e) => {
  if (e.code === "Space") {
    e.preventDefault();
    ticker ? pause() : start();
  }
});
render();
