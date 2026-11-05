import assert from 'node:assert/strict'
import {readFileSync,writeFileSync} from 'node:fs'
import * as frozen from '/Users/jonas/.codex-workspaces/worktrees/ios-36683f90-4afb-46b3-b505-f5d7b6d64528/weights/training/v4/game.ts'
import * as live from '/Users/jonas/.codex-workspaces/worktrees/ios-36683f90-4afb-46b3-b505-f5d7b6d64528/weights/src/game.ts'
const matches=JSON.parse(readFileSync('/tmp/rally-compact-audit/compact-independent-replays.json','utf8'))
const normalized=(x:unknown)=>JSON.parse(JSON.stringify(x))
let observationPairs=0,states=0
const signedZeroDifferences:any[]=[]
for(const [label,engine] of [['frozen-v4',frozen],['live-v4',live]] as const){
 for(const game of matches){
  let frame=engine.serve(game.seed)
  const loops=new engine.LoopDetector()
  assert.deepEqual(normalized(frame),normalized(game.initial))
  for(const row of game.steps){
   assert.equal(engine.endedRally(frame),row.scores===null)
   if(row.scores===null)frame=engine.serve(game.seed,frame)
   else{
    for(const side of [0,1] as const){
      const actual=Array.from(engine.observation(frame,side)),expected=row.observations[side]
      for(let i=0;i<6;i++)assert.ok(actual[i]===expected[i],`Nonzero observation mismatch at${game.seed}/${frame.tick}/${side}/${i}`)
      const differences=actual.map((v,i)=>Object.is(v,expected[i])?-1:i).filter(i=>i>=0)
      if(differences.length)signedZeroDifferences.push({engine:label,seed:game.seed,players:game.players,tick:frame.tick,side,actual,actual_negative_zero_indices:actual.map((v,i)=>Object.is(v,-0)?i:-1).filter(i=>i>=0),native_scalar_scores:row.scores[side],differences})
    }
    observationPairs++
    frame=engine.advance(frame,row.scores,loops)
   }
   assert.deepEqual(normalized(frame),normalized(row.state))
   states++
  }
  assert.ok(engine.endedMatch(frame))
 }
}
const path='/tmp/rally-compact-audit/compact-independent-verification.json'
const audit=JSON.parse(readFileSync(path,'utf8'))
audit.typescript={engines:['training/v4/game.ts','src/game.ts'],complete_matches_per_engine:matches.length,matched_transition_states:states,matched_observation_pairs:observationPairs,action_or_state_mismatches:0,signed_zero_observation_differences:signedZeroDifferences.length,observation_equality:'Exact numeric equality; signed zeros recorded separately for native inference parity'}
writeFileSync('/tmp/rally-compact-audit/compact-signed-zero-differences.json',JSON.stringify(signedZeroDifferences)+'\n')
writeFileSync(path,JSON.stringify(audit,null,2)+'\n')
console.log(JSON.stringify(audit.typescript))
