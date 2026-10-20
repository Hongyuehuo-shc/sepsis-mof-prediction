import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const out = process.argv[2];
const read = async (name) => JSON.parse(await fs.readFile(`${out}/${name}`, "utf8"));
const audit = await read("eicu_file_audit.json");
const predictors = await read("mimic_to_eicu_predictor_mapping.json");
const outcome = await read("eicu_cohort_outcome_mapping.json");
const notes = await read("eicu_parser_notes.json");
const summary = JSON.parse(await fs.readFile(`${out}/audit_summary.json`, "utf8"));
const wb = Workbook.create();
const font = "Arial";
const dark = "#1F4E78";
const mid = "#5B9BD5";

function tableSheet(name, title, data, widths, tabColor) {
  const s = wb.worksheets.add(name); s.showGridLines = false; s.tabColor = tabColor;
  s.getRange("A2").values = [[title]]; s.getRange("A2").format.font = {name:font,size:14,bold:true,color:"#1F1F1F"};
  const matrix = [data.headers, ...data.rows]; s.getRange("A5").write(matrix);
  const endRow = 5 + data.rows.length; const endCol = String.fromCharCode(64 + data.headers.length);
  s.getRange(`A5:${endCol}5`).format = {fill:dark,font:{name:font,size:10,bold:true,color:"#FFFFFF"},horizontalAlignment:"center",verticalAlignment:"center",wrapText:true,borders:{insideVertical:{style:"thin",color:"#FFFFFF"}}};
  s.getRange(`A6:${endCol}${endRow}`).format = {font:{name:font,size:9,color:"#222222"},verticalAlignment:"top",wrapText:true,borders:{insideHorizontal:{style:"thin",color:"#D9E2F3"}}};
  widths.forEach((width,index)=>s.getRangeByIndexes(0,index,endRow,1).format.columnWidth=width);
  s.getRange(`A5:${endCol}${endRow}`).format.rowHeight=30; s.getRange(`A5:${endCol}5`).format.rowHeight=40; s.freezePanes.freezeRows(5);
  return s;
}

const overview = wb.worksheets.add("Overview"); overview.showGridLines=false; overview.tabColor=dark;
overview.getRange("A2").values=[["eICU external-validation readiness audit"]]; overview.getRange("A2").format.font={name:font,size:14,bold:true};
overview.getRange("A4:B12").values=[
  ["Item","Result"],["Database","eICU-CRD v2.0"],["Official data tables",summary.official_tables_present],["Official SHA-256 matches",summary.sha256_matches],["Locked MIMIC model predictors",summary.model_predictors],["Numeric predictors",summary.numeric_predictors],["Categorical predictors",summary.categorical_predictors],["Ready to apply locked model","No"],["Required next step",summary.next_required_step]
];
overview.getRange("A4:B4").format={fill:dark,font:{name:font,size:10,bold:true,color:"#FFFFFF"},horizontalAlignment:"center"}; overview.getRange("A5:B12").format={font:{name:font,size:10},verticalAlignment:"top",wrapText:true,borders:{insideHorizontal:{style:"thin",color:"#D9E2F3"}}}; overview.getRange("A1:A12").format.columnWidth=31; overview.getRange("B1:B12").format.columnWidth=78; overview.getRange("B11").format={fill:"#FFF2CC",font:{name:font,size:10,bold:true,color:"#9C5700"}}; overview.getRange("B12").format.rowHeight=48;
overview.getRange("A15:B19").values=[["Interpretation",""],["Archive integrity","Complete: every official table matches the PhysioNet SHA-256 manifest"],["Predictor transportability","All 102 fitted model columns have an eICU source or derivation plan"],["Outcome transportability","Feasible, but the Sepsis-3 and hourly SOFA implementation must be validated before model evaluation"],["Key risk","Hospital-level structural missingness may reflect absent interfaces rather than absent clinical measurement"]]; overview.getRange("A15:B15").format={fill:mid,font:{name:font,size:10,bold:true,color:"#FFFFFF"}}; overview.getRange("A16:B19").format={font:{name:font,size:10},wrapText:true,verticalAlignment:"top",borders:{insideHorizontal:{style:"thin",color:"#D9E2F3"}}}; overview.getRange("A16:B19").format.rowHeight=36;

const p = tableSheet("Predictor mapping","Locked MIMIC CatBoost predictor mapping to eICU-CRD",predictors,[29,24,12,25,34,16,24,65],"#4472C4");
const o = tableSheet("Cohort and outcome","eICU Sepsis-3 cohort and incident MOF outcome mapping",outcome,[31,35,38,27,78],"#70AD47");
const f = tableSheet("File audit","Official archive integrity and schema audit",audit,[29,10,18,16,14,88,19],"#A5A5A5"); f.getRange(`C6:C${5+audit.rows.length}`).format.numberFormat="#,##0";
const n = tableSheet("Parser notes","Implementation risks and required handling",notes,[31,58,76,16],"#ED7D31");

wb.recalculate();
const check = await wb.inspect({kind:"table",range:"Overview!A1:B19",include:"values,formulas",tableMaxRows:20,tableMaxCols:3}); console.log(check.ndjson);
const mappingCheck = await wb.inspect({kind:"table",range:"Predictor mapping!A1:H16",include:"values,formulas",tableMaxRows:16,tableMaxCols:8}); console.log(mappingCheck.ndjson);
const errors = await wb.inspect({kind:"match",searchTerm:"#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",options:{useRegex:true,maxResults:200},summary:"final formula error scan"}); console.log(errors.ndjson);
for (const [sheetName,range,file] of [["Overview","A1:B19","overview_preview.png"],["Predictor mapping","A1:H20","mapping_preview.png"],["Cohort and outcome","A1:E14","outcome_preview.png"],["File audit","A1:G20","file_audit_preview.png"],["Parser notes","A1:D10","parser_notes_preview.png"]]) { const image=await wb.render({sheetName,range,scale:1.3}); await fs.writeFile(`${out}/${file}`,new Uint8Array(await image.arrayBuffer())); }
const file=await SpreadsheetFile.exportXlsx(wb); await file.save(`${out}/eicu_external_validation_audit_and_mapping.xlsx`);
