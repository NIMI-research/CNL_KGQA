import subprocess

def squall2sparql(input_final_path, output_file_path):
	input_list = []
	with open(input_final_path,"r") as file:
		lines = file.readlines()
	for line in lines:
		input_list.append(line)
	with open(output_file_path, 'w') as f:
		for x in input_list:
			x = x.strip("\n").strip()
			print(x)
			p = subprocess.Popen(['../tools/squall2sparql.sh',"-wikidata",x], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			out, err = p.communicate()
			if out.decode() == "":
				f.write(f"{err.decode()}\n")
				print(err.decode())
			else:
				f.write(f"{out.decode()}")
				print(out.decode().replace("\n",""))