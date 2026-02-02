import fs from "fs";
import path from "path";
import readline from "readline";
import { once } from "events";
import {
  buildAssetSeries,
  computeCrossForX,
  computePerAssetComputed,
  computeTargetsForXAtH,
  makeConstantCostModel,
  type Candle,
  type AssetSeries,
  type CrossForX,
  type PerAssetComputed,
  type TargetH,
  type RowMeta
} from "./types-formulas";

export type Schema = {
  meta_columns: string[];
  raw_market_columns: Record<string, string[]>;
  computed_feature_columns: Record<string, unknown>;
  target_columns_for_x: {
    horizons_bars: number[];
    materialized_targets: Record<string, string[]>;
  };
};

export type AssetKey = "btc" | "eth" | "sol" | "x";

export type AlignedCandles = {
  btc: Candle[];
  eth: Candle[];
  sol: Candle[];
  x: Candle[];
};

export type ColumnInfo = {
  columns: string[];
  perAssetComputed: Record<AssetKey, string[]>;
  crossColumns: string[];
  targetColumns: string[];
};

const REQUIRED_COLUMNS = [
  "open_time",
  "close_time",
  "open",
  "high",
  "low",
  "close",
  "volume",
  "quote_volume",
  "count",
  "taker_buy_volume",
  "taker_buy_quote_volume"
];

export async function loadSchema(schemaPath: string): Promise<Schema> {
  const content = await fs.promises.readFile(schemaPath, "utf8");
  return JSON.parse(content) as Schema;
}

export async function listCsvFiles(rootDir: string): Promise<string[]> {
  const results: string[] = [];

  async function walk(dir: string): Promise<void> {
    const entries = await fs.promises.readdir(dir, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        await walk(fullPath);
      } else if (entry.isFile() && entry.name.toLowerCase().endsWith(".csv")) {
        results.push(fullPath);
      }
    }
  }

  await walk(rootDir);
  results.sort();
  return results;
}

async function parseCsvFileToCandles(filePath: string): Promise<Candle[]> {
  const input = fs.createReadStream(filePath, { encoding: "utf8" });
  const rl = readline.createInterface({ input, crlfDelay: Infinity });

  let header: string[] | null = null;
  const index: Record<string, number> = {};
  const candles: Candle[] = [];

  for await (const rawLine of rl) {
    const line = rawLine.replace(/\r$/, "");
    if (!line) continue;

    if (!header) {
      header = line.split(",");
      header.forEach((col, idx) => {
        index[col] = idx;
      });
      const missing = REQUIRED_COLUMNS.filter(col => index[col] === undefined);
      if (missing.length > 0) {
        throw new Error(`Missing columns in ${filePath}: ${missing.join(", ")}`);
      }
      continue;
    }

    const values = line.split(",");
    if (values.length < header.length) continue;

    const candle: Candle = {
      ts_open_ms: Number(values[index.open_time]),
      ts_close_ms: Number(values[index.close_time]),
      open: Number(values[index.open]),
      high: Number(values[index.high]),
      low: Number(values[index.low]),
      close: Number(values[index.close]),
      volume: Number(values[index.volume]),
      quote_volume: Number(values[index.quote_volume]),
      count: Number(values[index.count]),
      taker_buy_volume: Number(values[index.taker_buy_volume]),
      taker_buy_quote_volume: Number(values[index.taker_buy_quote_volume])
    };

    candles.push(candle);
  }

  return candles;
}

export async function loadCandlesFromDir(dir: string): Promise<Candle[]> {
  const files = await listCsvFiles(dir);
  if (files.length === 0) {
    throw new Error(`No CSV files found under ${dir}`);
  }

  const all: Candle[] = [];
  for (const file of files) {
    const candles = await parseCsvFileToCandles(file);
    all.push(...candles);
  }

  all.sort((a, b) => a.ts_close_ms - b.ts_close_ms);
  return all;
}

export async function loadAndAlignCandles(paths: {
  btcDir: string;
  ethDir: string;
  solDir: string;
  xDir: string;
}): Promise<AlignedCandles> {
  const [btc, eth, sol, x] = await Promise.all([
    loadCandlesFromDir(paths.btcDir),
    loadCandlesFromDir(paths.ethDir),
    loadCandlesFromDir(paths.solDir),
    loadCandlesFromDir(paths.xDir)
  ]);

  return alignCandlesByCloseTime(btc, eth, sol, x);
}

export function alignCandlesByCloseTime(
  btc: Candle[],
  eth: Candle[],
  sol: Candle[],
  x: Candle[]
): AlignedCandles {
  const btcMap = new Map<number, Candle>(btc.map(c => [c.ts_close_ms, c]));
  const ethMap = new Map<number, Candle>(eth.map(c => [c.ts_close_ms, c]));
  const solMap = new Map<number, Candle>(sol.map(c => [c.ts_close_ms, c]));

  const aligned: AlignedCandles = { btc: [], eth: [], sol: [], x: [] };
  const xSorted = [...x].sort((a, b) => a.ts_close_ms - b.ts_close_ms);

  for (const xCandle of xSorted) {
    const btcCandle = btcMap.get(xCandle.ts_close_ms);
    const ethCandle = ethMap.get(xCandle.ts_close_ms);
    const solCandle = solMap.get(xCandle.ts_close_ms);
    if (!btcCandle || !ethCandle || !solCandle) continue;
    aligned.btc.push(btcCandle);
    aligned.eth.push(ethCandle);
    aligned.sol.push(solCandle);
    aligned.x.push(xCandle);
  }

  return aligned;
}

export function buildColumnInfo(schema: Schema, horizonBars: number): ColumnInfo {
  const columns: string[] = [];
  const perAssetComputed: Record<AssetKey, string[]> = {
    btc: [],
    eth: [],
    sol: [],
    x: []
  };
  const crossColumns: string[] = [];

  columns.push(...schema.meta_columns);

  for (const assetKey of Object.keys(schema.raw_market_columns)) {
    columns.push(...schema.raw_market_columns[assetKey]);
  }

  for (const groupKey of Object.keys(schema.computed_feature_columns)) {
    const value = schema.computed_feature_columns[groupKey];
    if (Array.isArray(value)) {
      crossColumns.push(...value);
      columns.push(...value);
      continue;
    }

    if (value && typeof value === "object") {
      const perAsset = value as Record<string, string[]>;
      for (const assetKey of Object.keys(perAsset)) {
        const normalized = assetKey.toLowerCase() as AssetKey;
        if (!perAssetComputed[normalized]) continue;
        perAssetComputed[normalized].push(...perAsset[assetKey]);
        columns.push(...perAsset[assetKey]);
      }
    }
  }

  const targetKey = `H=${horizonBars}`;
  const targetColumns = schema.target_columns_for_x?.materialized_targets?.[targetKey];
  if (!targetColumns) {
    throw new Error(`Schema does not define targets for ${targetKey}`);
  }
  columns.push(...targetColumns);

  return { columns, perAssetComputed, crossColumns, targetColumns };
}

function candleField(candle: Candle, field: string): number {
  const value = (candle as Record<string, number>)[field];
  if (value === undefined) {
    throw new Error(`Missing candle field ${field}`);
  }
  return value;
}

function computedField(comp: PerAssetComputed, field: string): number {
  const value = (comp as Record<string, number>)[field];
  if (value === undefined) {
    throw new Error(`Missing computed field ${field}`);
  }
  return value;
}

function crossField(cross: CrossForX, field: string): number {
  const value = (cross as Record<string, number>)[field];
  if (value === undefined) {
    throw new Error(`Missing cross field ${field}`);
  }
  return value;
}

function targetMap(target: TargetH, horizonBars: number): Record<string, number> {
  return {
    [`x_fwd_logret_${horizonBars}`]: target.x_fwd_logret_H,
    [`x_sigma_${horizonBars}`]: target.x_sigma_H,
    [`x_cost_${horizonBars}`]: target.x_cost_H,
    [`x_z_${horizonBars}`]: target.x_z_H,
    [`x_score_${horizonBars}`]: target.x_score_H
  };
}

function rowHasNonFinite(columns: string[], row: Record<string, number | string>): boolean {
  for (const col of columns) {
    const value = row[col];
    if (value === undefined) {
      throw new Error(`Missing value for column ${col}`);
    }
    if (typeof value === "number" && !Number.isFinite(value)) {
      return true;
    }
  }
  return false;
}

function formatValue(value: number | string): string {
  return typeof value === "number" ? value.toString() : value;
}

async function writeLine(stream: fs.WriteStream, line: string): Promise<void> {
  if (!stream.write(line)) {
    await once(stream, "drain");
  }
}

export async function writeFeatureCsv(options: {
  schema: Schema;
  aligned: AlignedCandles;
  tfMinutes: number;
  xSymbol: string;
  horizonBars: number;
  outputPath: string;
  exchange?: string;
  costBps?: number;
}): Promise<{ rowsWritten: number; rowsDropped: number }> {
  const {
    schema,
    aligned,
    tfMinutes,
    xSymbol,
    horizonBars,
    outputPath,
    exchange = "binance",
    costBps = 0
  } = options;

  const columnInfo = buildColumnInfo(schema, horizonBars);
  await fs.promises.mkdir(path.dirname(outputPath), { recursive: true });

  const output = fs.createWriteStream(outputPath, { encoding: "utf8" });
  await writeLine(output, `${columnInfo.columns.join(",")}\n`);

  const btcSeries: AssetSeries = buildAssetSeries(aligned.btc);
  const ethSeries: AssetSeries = buildAssetSeries(aligned.eth);
  const solSeries: AssetSeries = buildAssetSeries(aligned.sol);
  const xSeries: AssetSeries = buildAssetSeries(aligned.x);

  const costModel = makeConstantCostModel(costBps);

  let rowsWritten = 0;
  let rowsDropped = 0;

  for (let i = 0; i < aligned.x.length; i++) {
    const meta: RowMeta = {
      ts_open_ms: aligned.x[i].ts_open_ms,
      ts_close_ms: aligned.x[i].ts_close_ms,
      tf_minutes: tfMinutes,
      exchange,
      x_symbol: xSymbol
    };

    const btcComp = computePerAssetComputed(btcSeries, i);
    const ethComp = computePerAssetComputed(ethSeries, i);
    const solComp = computePerAssetComputed(solSeries, i);
    const xComp = computePerAssetComputed(xSeries, i);
    const cross = computeCrossForX(btcSeries, ethSeries, solSeries, xSeries, i);
    const target = computeTargetsForXAtH(xSeries, i, horizonBars, costModel);
    const targets = targetMap(target, horizonBars);

    const row: Record<string, number | string> = {
      ts_open_ms: meta.ts_open_ms,
      ts_close_ms: meta.ts_close_ms,
      tf_minutes: meta.tf_minutes,
      exchange: meta.exchange,
      x_symbol: meta.x_symbol
    };

    for (const assetKey of Object.keys(schema.raw_market_columns)) {
      const lower = assetKey.toLowerCase() as AssetKey;
      const candle =
        lower === "btc"
          ? aligned.btc[i]
          : lower === "eth"
          ? aligned.eth[i]
          : lower === "sol"
          ? aligned.sol[i]
          : aligned.x[i];

      for (const column of schema.raw_market_columns[assetKey]) {
        const field = column.slice(lower.length + 1);
        row[column] = candleField(candle, field);
      }
    }

    const perAssetMap: Record<AssetKey, PerAssetComputed> = {
      btc: btcComp,
      eth: ethComp,
      sol: solComp,
      x: xComp
    };

    for (const assetKey of Object.keys(columnInfo.perAssetComputed)) {
      const lower = assetKey as AssetKey;
      const comp = perAssetMap[lower];
      for (const column of columnInfo.perAssetComputed[lower]) {
        const field = column.slice(lower.length + 1);
        row[column] = computedField(comp, field);
      }
    }

    for (const column of columnInfo.crossColumns) {
      row[column] = crossField(cross, column);
    }

    for (const column of columnInfo.targetColumns) {
      row[column] = targets[column];
    }

    if (rowHasNonFinite(columnInfo.columns, row)) {
      rowsDropped += 1;
      continue;
    }

    const line = columnInfo.columns.map(col => formatValue(row[col])).join(",");
    await writeLine(output, `${line}\n`);
    rowsWritten += 1;
  }

  output.end();
  await once(output, "finish");

  return { rowsWritten, rowsDropped };
}
