import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const out = process.argv[2] || `${root}/results/12_manuscript_tables`;
const table1 = JSON.parse(await fs.readFile(`${out}/table1_data.json`, "utf8"));
const missing = JSON.parse(await fs.readFile(`${out}/missingness_data.json`, "utf8"));
const meta = JSON.parse(await fs.readFile(`${out}/table_metadata.json`, "utf8"));
const wb = Workbook.create();
const font = "Arial";

function buildSheet(name, title, data, widths) {
  const sheet = wb.worksheets.add(name); sheet.showGridLines = false; sheet.tabColor = name === "Table 1" ? "#1F4E78" : "#5B9BD5";
  sheet.getRange("A2").values = [[title]]; sheet.getRange("A2").format.font = {name:font,size:14,bold:true,color:"#1F1F1F"};
  sheet.getRange("A3:G3").format.borders = {bottom:{style:"thin",color:"#4472C4"}};
  const matrix = [data.headers, ...data.rows]; const endRow = 5 + data.rows.length; const endCol = String.fromCharCode(64 + data.headers.length);
  sheet.getRange("A5").write(matrix); const header = sheet.getRange(`A5:${endCol}5`); header.format = {fill:"#1F4E78",font:{name:font,size:10,bold:true,color:"#FFFFFF"},horizontalAlignment:"center",verticalAlignment:"center",wrapText:true,borders:{preset:"inside",style:"thin",color:"#FFFFFF"}};
  const body = sheet.getRange(`A6:${endCol}${endRow}`); body.format.font = {name:font,size:10,color:"#222222"}; body.format.verticalAlignment = "center"; body.format.borders = {insideHorizontal:{style:"thin",color:"#D9E2F3"}};
  sheet.getRange(`A6:A${endRow}`).format.horizontalAlignment = "left"; if (data.headers.length > 1) sheet.getRange(`B6:${endCol}${endRow}`).format.horizontalAlignment = "right";
  widths.forEach((w,i)=>sheet.getRangeByIndexes(0,i,endRow,1).format.columnWidth=w); sheet.getRange(`A5:${endCol}${endRow}`).format.rowHeight=20; header.format.rowHeight=32; sheet.freezePanes.freezeRows(5);
  return sheet;
}

const s1 = buildSheet("Table 1", "Baseline characteristics by sepsis-associated MOF status", table1, [39,22,22,22,18,12,10]);
s1.getRange(`A${7+table1.rows.length}`).values = [["Continuous variables are median [IQR]; categorical variables are n (%). P values use Mann–Whitney U, chi-square, or Fisher exact tests. SMD is the absolute standardized mean difference."]]; s1.getRange(`A${7+table1.rows.length}`).format = {font:{name:font,size:9,italic:true,color:"#595959"},wrapText:true}; s1.getRange(`A${7+table1.rows.length}`).format.rowHeight=36;
const s2 = buildSheet("Missingness", "Missingness of selected baseline variables", missing, [39,20,20,23,20]); const last2=5+missing.rows.length; s2.getRange(`C6:E${last2}`).format.numberFormat="0.0";
const methods = wb.worksheets.add("Methods"); methods.showGridLines=false; methods.tabColor="#A5A5A5"; methods.getRange("A2").values=[["Table definitions and cohort summary"]]; methods.getRange("A2").format.font={name:font,size:14,bold:true}; methods.getRange("A5:B12").values=[["Item","Definition"],["Cohort",`${meta.cohort_n} eligible Sepsis-3 ICU stays`],["MOF group",`${meta.mof_n} stays (${(meta.mof_prevalence*100).toFixed(1)}%)`],["Non-MOF group",`${meta.no_mof_n} stays`],["Continuous summary",meta.continuous_summary],["Categorical summary",meta.categorical_summary],["Statistical tests",`${meta.continuous_test}; ${meta.categorical_test}`],["SMD",meta.smd]]; methods.getRange("A5:B5").format={fill:"#1F4E78",font:{name:font,size:10,bold:true,color:"#FFFFFF"}}; methods.getRange("A6:B12").format.font={name:font,size:10}; methods.getRange("A1:A12").format.columnWidth=24; methods.getRange("B1:B12").format.columnWidth=64; methods.getRange("B6:B12").format.wrapText=true;
wb.recalculate();
const check = await wb.inspect({kind:"table",range:"Table 1!A1:G20",include:"values,formulas",tableMaxRows:20,tableMaxCols:7}); console.log(check.ndjson);
const errors = await wb.inspect({kind:"match",searchTerm:"#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",options:{useRegex:true,maxResults:100},summary:"final formula error scan"}); console.log(errors.ndjson);
const preview = await wb.render({sheetName:"Table 1",range:`A1:G${Math.min(30,8+table1.rows.length)}`,scale:1.5}); await fs.writeFile(`${out}/table1_preview.png`,new Uint8Array(await preview.arrayBuffer()));
const file = await SpreadsheetFile.exportXlsx(wb); await file.save(`${out}/manuscript_tables.xlsx`);
