import openai
import json
import pandas as pd


class OpenAIInference:
    def __init__(self, path_test_file_jsonl: str, path_to_save: str) -> None:
        """
        Initialize an instance of the OpenAIInference class.

        Args:
            path_test_file_jsonl (str): the path to the test file in JSONL format
            path_to_save (str): the path to save the predicted outputs in CSV format
        """
        self.test_path = path_test_file_jsonl
        self.output_path = path_to_save
        self.actual_output = []
        self.predicted_output =  []
        self.df = pd.read_json(self.test_path, lines=True)

    def _get_openai_model_output(self, model_sent: str, input_prompt: str) -> str:
        """
        Get the output of an OpenAI GPT model for a given prompt.

        Args:
            model_sent (str): the name of the OpenAI GPT model to use
            input_prompt (str): the input prompt for the GPT model

        Returns:
            str: the predicted output from the GPT model
        """
        first_model = model_sent
        prompt = input_prompt
        result = openai.Completion.create(model=first_model, prompt=prompt, max_tokens=100, temperature=0, top_p=1, n=1,
                                          stop=['\n'])
        return result['choices'][0]['text']


    def _get_inference_of_openai_model(self, model_name: str) -> None:
        """
        Get the predicted outputs for a given OpenAI GPT model.

        Args:
            model_name (str): the name of the OpenAI GPT model to use
        """
        for _, row in self.df.iterrows():
            self.actual_output.append(row["completion"])
            pred = self._get_openai_model_output(model_name, row['prompt'])
            self.predicted_output.append(pred)

        df = pd.DataFrame(zip(self.actual_output, self.predicted_output), columns=['GT', 'Predictions'])
        df.to_csv(f"{self.output_path}.csv")