from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import os

HUGGING_FACE_API_KEY = None

class MyLLM:
    def __init__(self):
        model_id = "google/flan-t5-large"
        llm = HuggingFacePipeline.from_model_id(
            model_id=model_id,
            device=0,
            task="text2text-generation",
            model_kwargs={"temperature": 1e-5, "max_length": 1000, "do_sample": True},
        )
        template = """{question}"""

        prompt = PromptTemplate(template=template, input_variables=["question"])

        self.llm_chain = LLMChain(prompt=prompt, llm=llm)

    def ask_question(self, question):
        result = self.llm_chain(question)
        return result['text']
    
    def gen_question(self, recommend, history):
        question = "Please rearrange the id_recommendation list according to the user's history interactions. The recommendation list and history (sorted from old to latest) will be presented below.\n"
        recommend_list = "The recommendation list is as follows, which have two items, representing the ids and the corresponding movie titles respectively: ids_recommendation=["
        for i, item in enumerate(recommend):
            recommend_list += "%d, " % (item[0])
        recommend_list = recommend_list[:-2]
        recommend_list += "], titles_recommendation=["
        for i, item in enumerate(recommend):
            recommend_list += "'%s', " % (item[1])
        recommend_list = recommend_list[:-2]
        recommend_list += "]."
        history_list = "\nThe history list is as follows, which have two items, representing the ids and the corresponding movie titles respectively: ids_history=["
        for i, item in enumerate(history):
            history_list += "%d, " % (item[0])
        history_list = history_list[:-2]
        history_list += "], titles_history=["
        for i, item in enumerate(history):
            history_list += "'%s', " % (item[1])
        history_list = history_list[:-2]
        history_list += "]."
        end_text = "\nPlease decide the rearranged id_recommendation list. You should include and only include the movie ids from the original list. You should at least perform one change."

        question = question + recommend_list + history_list + end_text
        
        return question

if __name__ == '__main__':
    my_llm = MyLLM()
    question = """
Please rearrange the movie recommendation list according to the user's history interactions. Specifically, move the movie that the user might want to watch to the front.
The recommendation list is as follows, which have two lists, representing the ids and the corresponding movie titles: [16, 27, 53, 0, 63,], ['Belly (1998)', 'Gremlins (1984)', 'New Rose Hotel (1998)', 'White Balloon, The (Badkonake Sefid ) (1995)', 'Mummy's Ghost, The (1944)'].
The history list is as follows, which have two lists, representing the ids and the corresponding movie titles: [69, 70, 71, 72, 73, 74, 75, 76, 77, 78,], ['From Dusk Till Dawn (1996)', 'Fair Game (1995)', 'Kicking and Screaming (1995)', 'Misérables, Les (1995)', 'Bed of Roses (1996)', 'Big Bully (1996)', 'Screamers (1995)', 'Nico Icon (1995)', 'Crossing Guard, The (1995)', 'Juror, The (1996)'].
Please answer with the list of movie ids (do not answer with titles, instead answer with their corresponding ids) that have been rearranged. All items in the recommendation list need to be included and only included once. You should at least perform three changes.    
"""
    print(my_llm.ask_question(question))

