#!/usr/bin/env node
"use strict";

const { spawn, execFileSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const PATHS_FILE = path.join(__dirname, ".paths.json");
const GUARD_ENV = "__ONECONTEXT_NPM_WRAPPER__";

// Check if a candidate binary resolves back to our own package (circular reference)
function isOwnBinary(candidatePath) {
  try {
    const realPath = fs.realpathSync(candidatePath);
    const pkgDir = fs.realpathSync(__dirname);
    if (realPath.startsWith(pkgDir + path.sep)) {
      return true;
    }
  } catch {
    // can't resolve, assume it's not ours
  }
  return false;
}

// Get npm bin directories to exclude (avoid recursion)
function npmBinDirs() {
  const dirs = new Set();
  try {
    dirs.add(require("child_process").execSync("npm bin -g", { encoding: "utf8" }).trim());
  } catch { /* ignore */ }
  try {
    dirs.add(require("child_process").execSync("npm bin", { encoding: "utf8" }).trim());
  } catch { /* ignore */ }
  try {
    dirs.add(require("child_process").execSync("npm prefix -g", { encoding: "utf8" }).trim() + "/bin");
  } catch { /* ignore */ }
  return dirs;
}

// Fallback: search PATH for command, excluding npm bin dirs and our own binaries
function searchPath(cmd) {
  const whichCmd = process.platform === "win32" ? "where" : "which";
  const whichArgs = process.platform === "win32" ? [cmd] : ["-a", cmd];
  const excludeDirs = npmBinDirs();
  try {
    const output = execFileSync(whichCmd, whichArgs, { encoding: "utf8" });
    const paths = output.split(/\r?\n/).map((p) => p.trim()).filter(Boolean);
    for (const p of paths) {
      // Check 1: skip npm bin directories
      const dir = path.dirname(p);
      let skip = false;
      for (const excl of excludeDirs) {
        if (dir === excl || p.startsWith(excl + path.sep)) {
          skip = true;
          break;
        }
      }
      // Check 2: resolve symlinks and skip if it points back to our package
      if (!skip && isOwnBinary(p)) {
        skip = true;
      }
      if (!skip) return p;
    }
  } catch {
    // not found
  }
  return null;
}

function resolve(cmd) {
  // 1. Try .paths.json
  if (fs.existsSync(PATHS_FILE)) {
    try {
      const data = JSON.parse(fs.readFileSync(PATHS_FILE, "utf8"));
      if (data[cmd]) {
        // Validate the cached path still exists and isn't our own binary
        if (fs.existsSync(data[cmd]) && !isOwnBinary(data[cmd])) {
          return data[cmd];
        }
      }
    } catch {
      // corrupted file, fall through
    }
  }

  // 2. Fallback: search PATH
  const found = searchPath(cmd);
  if (found) return found;

  // 3. Not found
  console.error(
    `[onecontext-ai] ERROR: Could not find "${cmd}" command.\n` +
    `Please install the Python package first:\n` +
    `  pip install aline-ai\n` +
    `Then re-run: npm rebuild onecontext-ai`
  );
  process.exit(1);
}

function run(cmd) {
  // Recursion guard: detect circular spawning via environment variable
  if (process.env[GUARD_ENV]) {
    console.error(
      `[onecontext-ai] ERROR: Circular reference detected.\n` +
      `The npm wrapper is calling itself instead of the Python binary.\n` +
      `Please install the Python package first:\n` +
      `  pip install aline-ai\n` +
      `Then re-run: npm rebuild onecontext-ai`
    );
    process.exit(1);
  }

  const binPath = resolve(cmd);
  const args = process.argv.slice(2);
  const child = spawn(binPath, args, {
    stdio: "inherit",
    env: { ...process.env, [GUARD_ENV]: "1" },
  });

  // Forward signals to child process
  for (const sig of ["SIGINT", "SIGTERM", "SIGHUP"]) {
    process.on(sig, () => {
      child.kill(sig);
    });
  }

  child.on("error", (err) => {
    console.error(`[onecontext-ai] Failed to start ${cmd}: ${err.message}`);
    process.exit(1);
  });

  child.on("exit", (code, signal) => {
    if (signal) {
      process.kill(process.pid, signal);
    } else {
      process.exit(code ?? 1);
    }
  });
}

module.exports = { run };
