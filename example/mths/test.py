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
                   key='sk-dKMuUHsISnfTuYGPDb78437190Db4bC19f968bC796260230',  # your key, e.g., 'sk-abcdefghijklmn'
                   model='gpt-5-mini',  # your llm, e.g., 'gpt-3.5-turbo' 'claude-3-5-sonnet-20240620'
                   timeout=180)

    response = llm.draw_sample("1+1 = ??")
    print(response)
    



if __name__ == '__main__':
    main()
