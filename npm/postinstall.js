#!/usr/bin/env node
"use strict";

const { execSync, execFileSync } = require("child_process");
const fs = require("fs");
const path = require("path");
const os = require("os");

const PACKAGE = "aline-ai";
const COMMANDS = ["onecontext", "oc"];
const PATHS_FILE = path.join(__dirname, ".paths.json");
const INSTALL_STATE_FILE = path.join(os.homedir(), ".aline", "install-state.json");

function log(msg) {
  console.log(`[onecontext-ai] ${msg}`);
}

function warn(msg) {
  console.warn(`[onecontext-ai] WARNING: ${msg}`);
}

// Detect available Python package manager: uv > pipx > pip3 > pip
function detectPkgManager() {
  const candidates = ["uv", "pipx", "pip3", "pip"];
  for (const cmd of candidates) {
    try {
      execFileSync(cmd, ["--version"], { stdio: "ignore" });
      return cmd;
    } catch {
      // not found, try next
    }
  }
  return null;
}

// Build install args for the given package manager
// Primary command tries to upgrade/install latest; fallback forces reinstall
function installArgs(mgr) {
  switch (mgr) {
    case "uv":
      return [["uv", "tool", "upgrade", PACKAGE], ["uv", "tool", "install", "--force", PACKAGE]];
    case "pipx":
      return [["pipx", "upgrade", PACKAGE], ["pipx", "install", "--force", PACKAGE]];
    case "pip3":
      return [["pip3", "install", "--upgrade", PACKAGE], ["pip3", "install", "--upgrade", "--force-reinstall", PACKAGE]];
    case "pip":
      return [["pip", "install", "--upgrade", PACKAGE], ["pip", "install", "--upgrade", "--force-reinstall", PACKAGE]];
    default:
      return [];
  }
}

// Run install, retry with force/upgrade on failure (2 minute timeout per attempt)
const INSTALL_TIMEOUT = 120_000;

function install(mgr) {
  const [primary, fallback] = installArgs(mgr);
  log(`Installing ${PACKAGE} via ${mgr}...`);
  try {
    execFileSync(primary[0], primary.slice(1), { stdio: "inherit", timeout: INSTALL_TIMEOUT });
    return true;
  } catch (err) {
    if (err.killed) {
      warn(`Installation timed out after ${INSTALL_TIMEOUT / 1000}s.`);
      return false;
    }
    log(`Retrying with ${fallback.slice(-1)[0]}...`);
    try {
      execFileSync(fallback[0], fallback.slice(1), { stdio: "inherit", timeout: INSTALL_TIMEOUT });
      return true;
    } catch (err2) {
      if (err2.killed) {
        warn(`Installation timed out after ${INSTALL_TIMEOUT / 1000}s.`);
      }
      return false;
    }
  }
}

// Get npm global/local bin directories to exclude from search (avoid recursion)
function npmBinDirs() {
  const dirs = new Set();
  try {
    dirs.add(execSync("npm bin -g", { encoding: "utf8" }).trim());
  } catch { /* ignore */ }
  try {
    dirs.add(execSync("npm bin", { encoding: "utf8" }).trim());
  } catch { /* ignore */ }
  // Also add the directory where this package's bins will be linked
  try {
    dirs.add(execSync("npm prefix -g", { encoding: "utf8" }).trim() + "/bin");
  } catch { /* ignore */ }
  return dirs;
}

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

// Discover absolute path for a command, filtering out npm bin directories and our own binaries
function discoverPath(cmd, excludeDirs) {
  const whichCmd = process.platform === "win32" ? "where" : "which";
  const whichArgs = process.platform === "win32" ? [cmd] : ["-a", cmd];
  try {
    const output = execFileSync(whichCmd, whichArgs, { encoding: "utf8" });
    const paths = output.split(/\r?\n/).map((p) => p.trim()).filter(Boolean);
    for (const p of paths) {
      const dir = path.dirname(p);
      let skip = false;
      for (const excl of excludeDirs) {
        if (dir === excl || p.startsWith(excl + path.sep)) {
          skip = true;
          break;
        }
      }
      // Also check via symlink resolution to catch edge cases
      if (!skip && isOwnBinary(p)) {
        skip = true;
      }
      if (!skip) return p;
    }
  } catch {
    // command not found
  }
  return null;
}

function fail(msg) {
  console.error(`[onecontext-ai] ERROR: ${msg}`);
  console.error(`[onecontext-ai] Please install manually:\n`);
  console.error(`  pip install aline-ai\n`);
  console.error(`Then re-run: npm rebuild onecontext-ai`);
  fs.writeFileSync(PATHS_FILE, JSON.stringify({}, null, 2));
  process.exit(1);
}

function writeInstallState(owner) {
  try {
    const normalizedOwner = String(owner || "").trim().toLowerCase() === "pip3" ? "pip" : String(owner || "").trim().toLowerCase();
    const payload = {
      owner: normalizedOwner,
      source: "npm_postinstall",
      updated_at: new Date().toISOString(),
      executable: "",
      python_executable: "",
    };
    fs.mkdirSync(path.dirname(INSTALL_STATE_FILE), { recursive: true });
    fs.writeFileSync(INSTALL_STATE_FILE, JSON.stringify(payload, null, 2) + "\n");
    log(`Recorded install owner '${payload.owner}' in ${INSTALL_STATE_FILE}`);
  } catch (err) {
    warn(`Failed to write install state file: ${err.message}`);
  }
}

function main() {
  const mgr = detectPkgManager();
  if (!mgr) {
    fail("No Python package manager found (uv, pipx, pip3, pip).");
  }

  const ok = install(mgr);
  if (!ok) {
    fail(`Failed to install ${PACKAGE} via ${mgr}.`);
  }

  // Discover command paths
  const excludeDirs = npmBinDirs();
  const paths = {};
  for (const cmd of COMMANDS) {
    const resolved = discoverPath(cmd, excludeDirs);
    if (resolved) {
      paths[cmd] = resolved;
      log(`Found ${cmd} at ${resolved}`);
    } else {
      warn(`Could not find ${cmd} in PATH (will search at runtime)`);
    }
  }

  fs.writeFileSync(PATHS_FILE, JSON.stringify(paths, null, 2));
  writeInstallState(mgr);

  log("");
  log("Installation complete. To finish setup, run:");
  log("");
  log("  onecontext init");
  log("");
}

main();
