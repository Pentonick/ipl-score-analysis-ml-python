import nbformat
from nbformat.v4 import new_notebook, new_code_cell

# 1. Naya notebook banayenge
nb = new_notebook()

# 2. Tumhari .txt file ko read karenge (Yahan apni txt file ka sahi naam likhna)
try:
    with open('hindi.txt', 'r', encoding='utf-8') as f:
        code_content = f.read()
    
    # 3. Code ko notebook ke cell me daalenge
    nb.cells.append(new_code_cell(code_content))

    # pip install nbformat
    # 4. .ipynb file ke roop me save karenge
    with open('IPL_Final_Project.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
    print("Badhai ho bhai! File successfully '.ipynb' me convert ho gayi hai. 😎")
    
except FileNotFoundError:
    print("Error: '.txt' file nahi mili. Naam check karo!")
