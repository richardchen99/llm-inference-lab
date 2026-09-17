import ts from 'typescript';
import { readFileSync } from 'node:fs';
const source = readFileSync(new URL('../src/model.ts', import.meta.url), 'utf8');
const { outputText } = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext } });
export const model = await import('data:text/javascript;base64,' + Buffer.from(outputText).toString('base64'));
