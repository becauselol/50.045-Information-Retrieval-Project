import wikienv, wrappers
import re
import requests
env = wikienv.WikiEnv()
#env = wrappers.FeverWrapper(env, split="dev")
env = wrappers.LoggingWrapper(env)

def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1
def clean_evidence_url(word):
    
    word = word.replace('-LRB-','')
    word = word.replace('-RRB-','')
    
    return word
    
def search_term(term,top_sentence=12,count= 0):
    if count >=2:
        print("Nothing found")
        return ''
    action = f'Search[{term}]'
    action[0].lower() + action[1:]
    print(action)
    res = step(env, action[0].lower() + action[1:])[0]
    
    if isinstance(res,str): ## Could not find but found similar term
        match = re.search(r"Similar: \[(.*?)\]", res)
        print(res)
        if match:
            list_str = match.group(1)
            # Split the list string and get the first element
            elements = re.findall(r"'(.*?)'", list_str)
            if elements:
                first_element = elements[0]
                print(f"Searching the most similar term {first_element} instead")
                count+=1
                return search_term(first_element,top_sentence,count)
    if isinstance(res,list):
        if res[0].startswith('There were no results matching the query'): ## Nothing found
            print("Nothing found")
            return ''
        
        return ''.join(res[:top_sentence])
    print("Unknown error",res)
    return ''
