// Exercise the real UI state handlers without a browser or third-party packages.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('hanhua/ui/app.js', 'utf8');
const handlers = source.slice(source.indexOf('const removedJobIds ='), source.indexOf('async function loadPages'));
const onEvent = source.slice(source.indexOf('function onEvent(e)'), source.indexOf('function onOpEvent(e)'));
const state = { jobs: new Map(), pages: new Map(), sel: null };
const context = { S: state, SHOT: true, VIEWS: {}, renderEngineCard() {} };
vm.runInNewContext(handlers + '\n' + onEvent + '\nglobalThis.api={onEvent,setJobs,upsertJob,removeJob};', context);
const {onEvent: event, setJobs} = context.api;
const a = { id: 'first', status: 'done', title: 'same-file' };
event({type:'job', job:a});
assert.equal(state.jobs.size, 1);
event({type:'removed', job_id:a.id});
assert.equal(state.jobs.size, 0);
event({type:'job', job:a}); // A delayed normal update must not resurrect a removed task.
assert.equal(state.jobs.size, 0);
event({type:'job', job:{id:'legacy', status:'done', removed:true}});
assert.equal(state.jobs.size, 0);
setJobs([a, {id:'legacy', removed:true}, {id:'second', title:'same-file', status:'queued'}]);
assert.equal(state.jobs.size, 1);
assert.equal(state.jobs.has('second'), true);
console.log('UI removal, delayed events, legacy tombstones and independent reimport: passed');
