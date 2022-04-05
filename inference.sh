# RUN : sh inference.sh
# results 내 파일들을 모두 지웁니다
# 해당 모델의 학습이 모두 끝날 경우 best_model 에 저장이 될 것이기 때문에
# results 내 파일들은 지워도 됩니다.

python main_infer.py \
--model klue/roberta-large \
--model_name refactor-test && rm -rf results 