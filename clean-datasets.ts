import fs from "fs";
import path from "path";
import readline from "readline";
import { once } from "events";

const DATASETS_DIR = path.resolve("Datasets");
const OUTPUT_DIR = path.resolve("Datasets_cleaned");

function normalizeSymbol(value: string): string {
  return value.trim().toUpperCase();
}

function deriveSymbolFromPath(filePath: string): string {
  const base = path.basename(filePath, ".csv");
  const match = base.match(/^(.*)USDT/i);
  if (match?.[1]) {
    return normalizeSymbol(match[1]);
  }

  const parts = filePath.split(path.sep).map(p => p.toUpperCase());
  const tokenIndex = parts.indexOf("TOKEN_X");
  if (tokenIndex >= 0 && parts[tokenIndex + 1]) {
    return normalizeSymbol(parts[tokenIndex + 1]);
  }

  for (const asset of ["BTC", "ETH", "SOL"]) {
    if (parts.includes(asset)) return asset;
  }

  throw new Error(`Unable to derive symbol for ${filePath}`);
}

async function listCsvFiles(rootDir: string): Promise<string[]> {
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

function buildHeader(columns: string[]): { header: string[]; openTimeIndex: number; ignoreIndex: number } {
  let openTimeIndex = -1;
  let ignoreIndex = -1;

  columns.forEach((col, idx) => {
    if (col === "open_time") openTimeIndex = idx;
    if (col === "ignore") ignoreIndex = idx;
  });

  if (openTimeIndex === -1) {
    throw new Error("CSV header missing open_time column");
  }
  if (ignoreIndex === -1) {
    throw new Error("CSV header missing ignore column");
  }

  const header: string[] = [];
  for (const col of columns) {
    if (col === "ignore") continue;
    if (col === "open_time") {
      header.push("open_time", "symbol");
      continue;
    }
    header.push(col);
  }

  return { header, openTimeIndex, ignoreIndex };
}

async function cleanFile(inputPath: string, outputPath: string): Promise<void> {
  const symbol = deriveSymbolFromPath(inputPath);
  await fs.promises.mkdir(path.dirname(outputPath), { recursive: true });

  const input = fs.createReadStream(inputPath, { encoding: "utf8" });
  const output = fs.createWriteStream(outputPath, { encoding: "utf8" });
  const rl = readline.createInterface({ input, crlfDelay: Infinity });

  let originalHeader: string[] | null = null;

  for await (const rawLine of rl) {
    const line = rawLine.replace(/\r$/, "");
    if (!line) continue;

    if (!originalHeader) {
      originalHeader = line.split(",");
      const { header } = buildHeader(originalHeader);
      output.write(`${header.join(",")}\n`);
      continue;
    }

    const values = line.split(",");
    if (values.length < originalHeader.length) {
      continue;
    }

    const cleaned: string[] = [];
    for (let i = 0; i < originalHeader.length; i++) {
      const col = originalHeader[i];
      if (col === "ignore") continue;
      if (col === "open_time") {
        cleaned.push(values[i], symbol);
        continue;
      }
      cleaned.push(values[i]);
    }
    output.write(`${cleaned.join(",")}\n`);
  }

  output.end();
  await once(output, "finish");
}

async function main(): Promise<void> {
  const csvFiles = await listCsvFiles(DATASETS_DIR);
  if (csvFiles.length === 0) {
    throw new Error(`No CSV files found under ${DATASETS_DIR}`);
  }

  let processed = 0;
  for (const filePath of csvFiles) {
    const relative = path.relative(DATASETS_DIR, filePath);
    const outputPath = path.join(OUTPUT_DIR, relative);
    await cleanFile(filePath, outputPath);
    processed += 1;
  }

  console.log(`Cleaned ${processed} file(s) to ${OUTPUT_DIR}`);
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
