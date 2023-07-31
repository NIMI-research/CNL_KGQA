import sys
from SPARQLWrapper import SPARQLWrapper, JSON

class SPARQLWrapperUtility:
    def __init__(self, predictions_queries, output_path, endpoint_url="https://query.wikidata.org/sparql"):
        self.endpoint_url = endpoint_url
        self.predictions_queries = predictions_queries
        self.output_path = output_path

    def update_queries(self, executable_queries):
        """
        Update the executable_queries list by wrapping the digit after 'pq:P1545' in double quotes.
        """
        updated_queries = []

        for row in executable_queries:
            words = row.split()
            updated_words = []

            for i, word in enumerate(words):
                if word == 'pq:P1545':
                    updated_words.append(word)
                elif word.isdigit() and words[i-1] == 'pq:P1545':
                    updated_words.append(f'"{word}"')
                else:
                    updated_words.append(word)

            updated_query = ' '.join(updated_words)
            updated_queries.append(updated_query)

        return updated_queries

    def get_results(self, query):
        """
        Send a SPARQL query to the specified endpoint and return the results in JSON format.
        """
        user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
        sparql = SPARQLWrapper(self.endpoint_url, agent=user_agent)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        return sparql.query().convert()

    def wrapper(self, queries):
        """
        Process the given queries, obtain the results from the SPARQL endpoint, and return the results.
        """
        full_results = []
        
        for i, row in enumerate(queries):
            try:
                results = self.get_results(row)
                full_results.append((i, results["results"]["bindings"]))
            except Exception as e:
                full_results.append((i, 'exception error'))

        return full_results

    def execute(self):
        updated_queries = self.update_queries(self.predictions_queries)
        result_list = self.wrapper(updated_queries)

        with open(f'{self.output_path}.json', 'w') as f:
            json.dump(result_list, f)
