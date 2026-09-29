import sys

sys.path.append('../../')  # This is for finding all the modules

from evaluation_cvrp import CVRPEvaluation
from evaluation_tsp import TSPEvaluation
from evaluation_fssp import FSSPEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.mteop import MTAHD,EoHProfiler
from llm4ad.tools.profiler import ProfilerBase


def main():
    llm = HttpsApi(host='api.bltcy.ai',  # your host endpoint, e.g., 'api.openai.com', 'api.deepseek.com'
                   key='xxx',  # your key, e.g., 'sk-abcdefghijklmn'
                   model='gpt-5-mini',  # your llm, e.g., 'gpt-3.5-turbo' 'claude-3-5-sonnet-20240620'
                   timeout=240)

    # response = llm.draw_sample("1+1 = ??")
    # print(response)
    # input()

    task_list = []
    task_list.append(TSPEvaluation(timeout_seconds=1280))
    task_list.append(CVRPEvaluation(timeout_seconds=1280))
    task_list.append(FSSPEvaluation(timeout_seconds=1280))

    method = MTAHD(llm=llm,
                 profiler=EoHProfiler(log_dir='logs/eop', log_style='simple'),
                 evaluations=task_list,
                 max_sample_nums=1000,
                 max_generations=1000,
                 high_level_pop_size=8,
                 low_level_pop_size=2,
                 low_level_max_samples_each_program=4,
                 num_samplers=8,
                 num_evaluators=8,
                 enable_knowledge_transfer=True,
                 enable_low_level_search=True,
                 enable_pareto_population_management=True,
                   metaheuristic_type=2,
                 debug_mode=False)

    method.run()


if __name__ == '__main__':
    main()
