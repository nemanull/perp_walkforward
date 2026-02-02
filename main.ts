import fs from "fs";
import path from "path";
import {
  loadSchema,
  loadAndAlignCandles,
  writeFeatureCsv
} from "./compute-ops";

type CliArgs = {
  symbol: string;
  tfMinutes: number;
  horizonArg: number;
  costBps: number;
};

function printUsage(): void {
  console.log(
    [
      "Usage:",
      "  pnpm run build -- --HYPE --5 --4",
      "  node --loader ts-node/esm main.ts --symbol HYPE --tf 5 --horizon 20",
      "  node --loader ts-node/esm main.ts --symbol=HYPE --tf=5 --horizon=20 --cost-bps=0",
      "",
      "Notes:",
      "  - Horizon is in minutes unless it matches a schema bar horizon (4/12/36/72).",
      "  - Horizon minutes must be divisible by tf_minutes."
    ].join("\n")
  );
}

function parseArgs(argv: string[]): CliArgs {
  let symbol: string | null = null;
  let tfMinutes: number | null = null;
  let horizonArg: number | null = null;
  let costBps = 0;

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (!arg.startsWith("--")) continue;

    const [flag, inlineValue] = arg.slice(2).split("=");
    const nextValue =
      inlineValue ??
      (i + 1 < argv.length && !argv[i + 1].startsWith("--") ? argv[++i] : undefined);

    if (flag === "help") {
      printUsage();
      process.exit(0);
    }

    if (flag === "symbol") {
      if (!nextValue) throw new Error("Missing value for --symbol");
      symbol = nextValue.toUpperCase();
      continue;
    }
    if (flag === "tf") {
      if (!nextValue) throw new Error("Missing value for --tf");
      tfMinutes = Number(nextValue);
      continue;
    }
    if (flag === "horizon") {
      if (!nextValue) throw new Error("Missing value for --horizon");
      horizonArg = Number(nextValue);
      continue;
    }
    if (flag === "cost-bps") {
      if (!nextValue) throw new Error("Missing value for --cost-bps");
      costBps = Number(nextValue);
      continue;
    }

    if (/^\d+$/.test(flag)) {
      const numeric = Number(flag);
      if (tfMinutes === null) {
        tfMinutes = numeric;
      } else if (horizonArg === null) {
        horizonArg = numeric;
      } else {
        throw new Error(`Unexpected numeric flag --${flag}`);
      }
      continue;
    }

    if (!symbol) {
      symbol = flag.toUpperCase();
      continue;
    }

    throw new Error(`Unrecognized argument --${flag}`);
  }

  if (!symbol || !tfMinutes || !horizonArg) {
    printUsage();
    throw new Error("Missing required arguments. Expect --SYMBOL --TF --HORIZON.");
  }

  if (!Number.isFinite(tfMinutes) || tfMinutes <= 0) {
    throw new Error("tf_minutes must be a positive number.");
  }
  if (!Number.isFinite(horizonArg) || horizonArg <= 0) {
    throw new Error("horizon must be a positive number.");
  }

  return { symbol, tfMinutes, horizonArg, costBps };
}

function resolveHorizonBars(
  horizonArg: number,
  tfMinutes: number,
  allowedBars: number[]
): number {
  if (horizonArg % tfMinutes === 0) {
    const bars = horizonArg / tfMinutes;
    if (allowedBars.includes(bars)) {
      return bars;
    }
  }

  if (allowedBars.includes(horizonArg)) {
    return horizonArg;
  }

  const allowed = allowedBars.join(", ");
  if (horizonArg % tfMinutes !== 0) {
    throw new Error(
      `Horizon ${horizonArg} is not divisible by tf_minutes ${tfMinutes} and does not match schema bars (${allowed}).`
    );
  }

  const bars = horizonArg / tfMinutes;
  throw new Error(
    `Horizon ${horizonArg} minutes = ${bars} bars, which is not in schema (${allowed}).`
  );
}

function assertDirExists(dir: string): void {
  if (!fs.existsSync(dir)) {
    throw new Error(`Missing directory: ${dir}`);
  }
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));

  const schemaPath = path.resolve("schema_v1_polished.json");
  const schema = await loadSchema(schemaPath);
  const horizonBars = resolveHorizonBars(
    args.horizonArg,
    args.tfMinutes,
    schema.target_columns_for_x.horizons_bars
  );

  const cleanedRoot = path.resolve("Datasets_cleaned");
  const tfDir = `${args.tfMinutes}MIN`;

  const btcDir = path.join(cleanedRoot, "BTC", tfDir);
  const ethDir = path.join(cleanedRoot, "ETH", tfDir);
  const solDir = path.join(cleanedRoot, "SOL", tfDir);
  const xDir = path.join(cleanedRoot, "TOKEN_X", args.symbol, tfDir);

  [btcDir, ethDir, solDir, xDir].forEach(assertDirExists);

  const aligned = await loadAndAlignCandles({ btcDir, ethDir, solDir, xDir });
  if (aligned.x.length === 0) {
    throw new Error("No aligned rows found. Check cleaned data alignment.");
  }

  const outputDir = path.resolve("Outputs");
  const outputPath = path.join(
    outputDir,
    `${args.symbol}_${args.tfMinutes}m_H${horizonBars}.csv`
  );

  const result = await writeFeatureCsv({
    schema,
    aligned,
    tfMinutes: args.tfMinutes,
    xSymbol: args.symbol,
    horizonBars,
    outputPath,
    costBps: args.costBps
  });

  console.log(
    `Wrote ${result.rowsWritten} rows to ${outputPath} (dropped ${result.rowsDropped}).`
  );
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});









