const RUN_DIR = /(?:\/[^\s`]+\/)?runs\/[^\/\s`]+\/[^\/\s`]+/g
const ENCODER = new TextEncoder()

function normalizeRunDir(directory, value) {
  if (value.startsWith("/")) return value
  return `${directory}/${value}`
}

export const HieraGuard = async ({ client, directory }) => {
  const sessionRuns = new Map()
  const taskAgents = new Map()

  return {
    "tool.execute.before": async (input, output) => {
      if (input.tool !== "task") return
      const args = output.args || {}
      const prompt = String(args.prompt || "")
      const matches = prompt.match(RUN_DIR) || []
      if (matches.length) sessionRuns.set(input.sessionID, normalizeRunDir(directory, matches[0]))

      const agent = String(args.subagent_type || "")
      taskAgents.set(input.callID, agent)
      const result = Bun.spawnSync(
        ["python", `${directory}/tools/harness_guard.py`, "--opencode-check", agent],
        { stdin: ENCODER.encode(prompt), stdout: "pipe", stderr: "pipe" },
      )
      if (result.exitCode === 3) {
        taskAgents.delete(input.callID)
        const reason = result.stdout.toString().trim()
        throw new Error(`HieraResearch delegation guard: ${reason}`)
      }
      if (result.exitCode !== 0) {
        taskAgents.delete(input.callID)
        throw new Error(`HieraResearch delegation guard failed: ${result.stderr.toString().trim()}`)
      }
    },

    "tool.execute.after": async (input, output) => {
      if (input.tool !== "task") return
      const agent = taskAgents.get(input.callID) || String(input.args?.subagent_type || "")
      taskAgents.delete(input.callID)
      if (!agent) return
      const result = Bun.spawnSync(
        ["python", `${directory}/tools/harness_guard.py`, "--compact-result", agent],
        { stdin: ENCODER.encode(String(output.output || "")), stdout: "pipe", stderr: "pipe" },
      )
      if (result.exitCode !== 0) {
        throw new Error(`HieraResearch receipt guard failed: ${result.stderr.toString().trim()}`)
      }
      output.output = result.stdout.toString().trim()
    },

    event: async ({ event }) => {
      if (event.type !== "session.idle") return
      const sessionID = event.properties.sessionID
      const runDir = sessionRuns.get(sessionID)
      if (!runDir) return
      const result = Bun.spawnSync(
        ["python", `${directory}/tools/harness_watch.py`, "--check-run-idle", "--run-dir", runDir],
        { stdout: "pipe", stderr: "pipe" },
      )
      if (result.exitCode !== 3) return
      await client.tui.showToast({
        body: {
          title: "HieraResearch run still active",
          message: result.stdout.toString().trim(),
          variant: "warning",
          duration: 10000,
        },
        query: { directory },
      })
    },
  }
}
