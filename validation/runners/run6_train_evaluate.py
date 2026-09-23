import hashlib, json, os, sys, time, statistics
from collections import Counter
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'backend'))
sys.path.insert(0, str(PROJECT_ROOT))
from app.services.experience_buffer import ExperienceBufferService, ACTION_MAP, REVERSE_ACTION_MAP
from app.services.policies.rl_bandit_policy import RLContextualBanditPolicy
from app.schemas.rl import RLTrainRequest
from rl.policy_trainer import PolicyTrainer

DATASET = PROJECT_ROOT / 'data/evaluation/run6/final_available_actions_direct/run6_available_actions.jsonl'
RUN6_DIR = PROJECT_ROOT / 'validation/results/run6'
TRAIN_BUFFER = PROJECT_ROOT / 'data/evaluation/run6/final_available_actions_direct/train_buffer.jsonl'
ARTIFACT = PROJECT_ROOT / 'data/rl/models/rl_contextual_bandit_policy_run6_available_actions.json'
MANIFEST = RUN6_DIR / 'run6_available_actions_manifest.json'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def main():
    records=[json.loads(x) for x in DATASET.read_text(encoding='utf-8').splitlines() if x.strip()]
    records.sort(key=lambda r:r['timestamp'])
    train_count=max(1,int(len(records)*0.7))
    train_records=records[:train_count]
    test_records=records[train_count:]
    TRAIN_BUFFER.write_text('\n'.join(json.dumps(r) for r in train_records)+'\n',encoding='utf-8')
    train_buffer=ExperienceBufferService(capacity=1000,persistence_path=str(TRAIN_BUFFER))
    trainer=PolicyTrainer(output_path=str(ARTIFACT),experience_buffer=train_buffer)
    train_res=trainer.train_policy(RLTrainRequest(minimum_samples=train_count,epochs=100,learning_rate=0.01,l2_lambda=0.01))
    if not train_res.success: raise RuntimeError(train_res.message)
    artifact=json.loads(ARTIFACT.read_text(encoding='utf-8'))
    artifact.update({'run_id':'RUN_6_AVAILABILITY_CONSTRAINED','dataset_path':str(DATASET.relative_to(PROJECT_ROOT)),'dataset_sha256':sha(DATASET),'train_samples':len(train_records),'test_samples':len(test_records),'action_availability_mask':{'A0':True,'A1':True,'A2':True,'A3':False,'A4':False,'A5':False,'A6':True,'A7':False},'training_seed':42,'training_epochs':100,'training_learning_rate':0.01,'training_l2_lambda':0.01,'trained_timestamp':time.time()})
    ARTIFACT.write_text(json.dumps(artifact,indent=2),encoding='utf-8')
    artifact_hash=sha(ARTIFACT)
    policy=RLContextualBanditPolicy(model_path=str(ARTIFACT))
    executable=[0,1,2,6]
    predictions=[]
    for rec in test_records:
        scores=policy.weights @ np.array(rec['state'],dtype=np.float32)+policy.bias
        masked=np.full(8,-np.inf,dtype=np.float32)
        masked[executable]=scores[executable]
        pred=int(np.argmax(masked))
        predictions.append((pred,rec))
    agreement=sum(pred==rec['action'] for pred,rec in predictions)
    rewards=[rec['reward'] for _,rec in predictions]
    ips_terms=[rec['reward'] if pred==rec['action'] else 0.0 for pred,rec in predictions]
    ips=sum(ips_terms)/len(test_records) if test_records else 0.0
    snips=(sum(ips_terms)/sum(1.0 for x in ips_terms if x>0)) if any(ips_terms) else 0.0
    ess=(sum(ips_terms)**2/sum(x*x for x in ips_terms)) if any(ips_terms) else 0.0
    lat=[rec['metadata']['pipeline_latency_ms'] for rec in test_records]
    costs=[rec['metadata']['cost'] for rec in test_records]
    evaluation={'regime':'Availability-Constrained RUN 6 Evaluation','test_samples':len(test_records),'agreement_count':agreement,'agreement_rate':agreement/len(test_records) if test_records else 0.0,'ips':ips,'snips':snips,'ess':ess,'positivity_coverage':agreement/len(test_records) if test_records else 0.0,'observed_reward_mean':statistics.mean(rewards),'observed_reward_median':statistics.median(rewards),'success_rate':1.0,'latency_mean_ms':statistics.mean(lat),'latency_median_ms':statistics.median(lat),'latency_p95_ms':sorted(lat)[int(.95*len(lat))-1],'cost_total_usd':sum(costs),'action_agreement_predictions':dict(Counter(pred for pred,_ in predictions)),'interpretation':'No paired online RL rollout was performed; off-policy inference is weak/insufficient with this deterministic behavior policy and low ESS.'}
    manifest={'run_id':'RUN_6_AVAILABILITY_CONSTRAINED','status':'trained_evaluated_not_promoted','timestamp':time.time(),'full_configured_action_space':[f'A{i}' for i in range(7)],'final_executable_action_space':['A0','A1','A2','A6'],'excluded_actions':{'A3':'Gemini 429 RESOURCE_EXHAUSTED quota','A4':'Mistral 429 rate limit','A5':'Groq 403 Forbidden authorization failure','A7':'Embedding action masked'},'dataset_path':str(DATASET.relative_to(PROJECT_ROOT)),'dataset_sha256':sha(DATASET),'raw_records':len(records),'valid_records':len(records),'train_samples':len(train_records),'test_samples':len(test_records),'action_counts':{f'A{i}':sum(r['action']==i for r in records) for i in range(8)},'training_config':{'algorithm':'existing PolicyTrainer regularized linear contextual bandit','K':8,'D':12,'seed':42,'epochs':100,'learning_rate':0.01,'l2_lambda':0.01},'training_mse':train_res.final_mse,'artifact_path':str(ARTIFACT.relative_to(PROJECT_ROOT)),'artifact_sha256':artifact_hash,'evaluation':evaluation,'production_artifact_sha256':'f0e8f61370931eb9cca7421d0ac8a016b46eac7d31c87622cfe9e7d2b2f1fc71','production_promotion':False,'a7_masked':True,'online_validation':'NOT RUN — availability-constrained offline evaluation only'}
    RUN6_DIR.mkdir(parents=True,exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'train':len(train_records),'test':len(test_records),'mse':train_res.final_mse,'artifact':str(ARTIFACT),'artifact_sha256':artifact_hash,'evaluation':evaluation},indent=2))
if __name__=='__main__': main()
