import json
import re
import pandas as pd
import requests


class parser:
    """
    Class to parse the query generated by the squall2sparql tool.
    """

    def __init__(self, output_txt_file_from_tool,test_file_path,path_to_load_props):
        self.output_txt_file_from_tool = output_txt_file_from_tool
        self.test_file_path = test_file_path
        self.load_dict_path_prop = path_to_load_props
        self.prop = self._load_prop()


    def _get_ids_of_props_entities(self,ent_prop, types):
        """
        Function to get the ids of properties and entites given the labels of each.
        """

        params = dict(
            action='wbsearchentities',
            format='json',
            language='en',
            uselang='en',
            type=types,
            search=ent_prop
        )

        response = requests.get('https://www.wikidata.org/w/api.php?', params).json()
        responses = []
        if response is not None:
            for response in response.get('search'):
                if response.get('label') is not None and response.get('label') == ent_prop:
                    responses.append(response['id'])

        return responses


    def _load_prop(self):
        """
        Function to load dict which maps labels to ids..
        """

        with open(self.load_dict_path_prop,"r") as file:
            return json.load(file)



    def _parser_intermediate_sparql(self):
        """
        Function which converts the tool intermediate sparql to an executable wikidata sparql query.
        """

        queries = []
        ex_indices = []
        list_work= []
        with open(self.output_txt_file_from_tool, 'r') as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                if 'SELECT' in line or 'ASK' in line:
                    queries.append(line.replace("\n", ''))
                    ex_indices.append(int((i) ))
                    ques = re.findall(r'Question:\s(.*?)(?=\n|$)', eval_data_test[i]['prompt'])
                    list_work.append(ques)

        for query in queries:
            query = query.replace("  ", " ").strip()
            if '<publication_year>' in query:
                query = query.replace('<publication_year>', '<publication_date>')

        ent_list = []
        executable_queries = []
        
        
        with open(self.test_file_path, 'r') as f:
            mintaka_test = json.load(f)
            squall_test = mintaka_test['SQUALL']

            for index, row in enumerate(squall_test):
                pmt = row['prompt'].split("\n")
                if index in ex_indices:
                    new_list = re.findall(r"Q[0-9]*",pmt[2].strip())
                    ent_list.append(new_list)
                    del new_list

        assert len(ent_list) == len(ex_indices) == len(queries)
        list_props = ["p<", "ps<", "pq<", "n1<", "n2<"]
        for i, query in enumerate(queries):
            m = re.search(r"wdt:P31 \<([a-z_]+)\>", query)
            if m is not None:
                ent_prop = " ".join(m.group(1).split("_")).strip()
                if len(ent_prop) > 0:
                    ents = self._get_ids_of_props_entities(ent_prop, 'item')
                if len(ents) > 0:
                    query = query.replace(f'<{m.group(1)}>', f'wd:{ents[0]}')
            splits = query.split(" ")
            for word in splits:
                if word.startswith("<") and ">" in word and not any(e in word for e in list_props):
                    word = word.replace("<", "").replace(">", "")
                    word = " ".join(word.split("_")).strip()
                    if word is not None:
                        ets = self._get_ids_of_props_entities(word, 'item')
                        if len(ets) == 0:
                            ets = self._get_ids_of_props_entities("".join(word.split()), 'item')
                        qid = ''
                        for e in ets:
                            if e in ent_list[i]:
                                qid = e

                                break
                            else:
                                qid = ets[0]
                        query = query.replace(word.replace(" ", "_").strip(), f'wd:{qid}')
                elif word.startswith("p") and "<" in word:
                    idx1 = [pos for pos, char in enumerate(word) if char == "<"]
                    idx2 = [pos for pos, char in enumerate(word) if char == ">"]
                    ets = self.prop.get(word[idx1[0] + 1:idx2[0]])
                    if ets is not None:
                        query = query.replace((word[idx1[0] + 1:idx2[0]]), ets)
            query = query.replace("p<", "p:").replace("ps<", "ps:").replace("pq<", "pq:").replace("<w", "w")
            list_query = list(query)
            for idx in range(1, len(list_query) - 1):
                if list_query[idx - 1] != " " and list_query[idx] == ">" or list_query[idx + 1] != " " and list_query[idx] == "<":
                    list_query[idx] = ""
            query = "".join(list_query)
            executable_queries.append(query)
        return executable_queries, list_work