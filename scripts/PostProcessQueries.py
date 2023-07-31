import requests
import json
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd


class PostProcessing:
    """
    class for postproessing the intermediate queries from the model to make them executable.
    """
    def __init__(self, path_to_id_to_prop_dict, path_to_prop_dict, path_to_csv):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.label_list = self._make_label_list(self._load_id_to_prop())
        self.labels_sim = self.model.encode(self.label_list)
        self.path_to_id_to_prop_dict = path_to_id_to_prop_dict
        self.path_to_prop_dict = path_to_prop_dict
        self.prop = self._load_prop()
        self.path_to_csv = path_to_csv


    def _make_label_list(self, id_to_prop):
        """
        Function to create list from which most similar label is Fetched.
        """
        label_list = list(id_to_prop.values())
        return label_list


    def _load_id_to_prop(self):
        """
        Function to load dict for ids to label of property matching.
        """
        with open(self.path_to_id_to_prop_dict,"r") as file:
            return json.load(file)


    def _load_prop(self):
        """
         Function to load dict for label to ids of property matching.
         """

        with open(self.path_to_prop_dict,"r") as file:
            return json.load(file)


    def _cos_sim(self, element, model, labels_sim):

        """
         Function which gives the most similar label index for the hallucinated label..
         """
        x = model.encode([element])
        res = (util.dot_score(x, labels_sim))
        res = res.squeeze()
        idx = np.unravel_index(np.argmax(res, axis=None), res.shape)
        return idx[0]



    def _get_prediction_files(self,path):
        """
         Function to load the predictions CSV from the model.
         """

        df = pd.read_csv(path)
        post_process_sparql = []
        post_process_sparql_t = []
        for idx, row in df.iterrows():
            post_process_sparql.append(row["GT"])
            post_process_sparql_t.append(row["Predictions"])
        return post_process_sparql, post_process_sparql_t



    def _post_processing_sparklis(self, ps):
        """
         Function to Post process the intermediate sparklis query generated by the model.
         """

        processed = []
        for row in ps:
            splits = row.split()
            for split in splits:
                if ("<" in split or ">" in split) and ":" in split:
                    pid, label = split.replace("<", "").replace(">", "").strip().split(":", 1)
                    if self.prop.get(label) is not None:
                        row = row.replace(split, f"{label.replace('_', ' ')}")
                    else:
                        idx = self._cos_sim(label, self.model, self.labels_sim)
                        corrected_label = self.label_list[idx]
                        row = row.replace(split, f"{corrected_label.replace('_', ' ')}")
            processed.append(row.strip())

        return processed


    def _post_process_squall(self, post_process):
        """
         Function to Post process the intermediate squall query generated by the model.
         """

        processed = []
        for i, row in enumerate(post_process):
            splits = row.strip().replace("\n", "").split()
            for split in splits:
                if "<" in split and ">" in split and ":" in split:
                    if "is-" in split:
                        x = split.replace("is-", "").replace("<", "").replace(">", "").replace("?", "")
                        pid, label = x.split(":", 1)
                        if self.prop.get(label) is not None:
                            row = row.replace(split, f"is-<{label}>")
                        else:
                            idx = self._cos_sim(label, self.model, self.labels_sim)  
                            true_label = self.label_list[idx]
                            row = row.replace(split, f"is-<{true_label}>")
                    else:
                        x = split.replace("<", "").replace(">", "").replace("?", "")
                        pid, label = x.split(":", 1)
                        if self.prop.get(label) is not None:
                            row = row.replace(split, f"<{label}>")
                        else:
                            idx = self._cos_sim(label, self.model, self.labels_sim) 
                            true_label = self.label_list[idx]
                            row = row.replace(split, f"<{true_label}>")
            row = " ".join([e.strip() for e in row.split()]).strip()
            row = f'{row}?' if "?" not in row else row
            processed.append(row)
        return processed


    def _post_processing_sparql(self, post_process):
        """
         Function to Post process the intermediate sparql query generated by the model.
         """

        processed = []
        for row in post_process:
            splits = row.strip().split()
            for split in splits:
                if "<" in split and ">" in split and "/" not in split:
                    if "p:" in split:
                        x = split.replace("p:", "").replace("<", "").replace(">", "")
                        pid, label = x.split(":", 1)
                        if self.prop.get(label):
                            row = row.replace(split, f"p:{self.prop.get(label)}")
                        else:
                            idx = self._cos_sim(label, self.model, self.labels_sim)
                            similarity_id = self.prop.get(self.label_list[idx])
                            row = row.replace(split, f"p:{similarity_id}")
                    if "ps:" in split:
                        x = split.replace("ps:", "").replace("<", "").replace(">", "")
                        pid, label = x.split(":", 1)
                        if self.prop.get(label):
                            row = row.replace(split, f"ps:{self.prop.get(label)}")
                        else:
                            idx = self._cos_sim(label, self.model, self.labels_sim)
                            similarity_id = self.prop.get(self.label_list[idx])
                            row = row.replace(split, f"ps:{similarity_id}")
                    if "n1" in split:
                        x = split.replace("n1", "").replace("<", "").replace(">", "")
                        pid, label = x.split(":", 1)
                        if self.prop.get(label):
                            row = row.replace(split, f"n1:{self.prop.get(label)}")
                        else:
                            idx = self._cos_sim(label, self.model, self.labels_sim)
                            similarity_id = self.prop.get(self.label_list[idx])
                            row = row.replace(split, f"n1:{similarity_id}")
                    if "n2" in split:
                        x = split.replace("n2", "").replace("<", "").replace(">", "")
                        pid, label = x.split(":", 1)
                        if self.prop.get(label):
                            row = row.replace(split, f"n2:{self.prop.get(label)}")
                        else:
                            idx = self._cos_sim(label, self.model, self.labels_sim)
                            similarity_id = self.prop.get(self.label_list[idx])
                            row = row.replace(split, f"n2:{similarity_id}")
            row = " ".join(e.strip() for e in row.split()).strip()
            processed.append(row)
        return processed