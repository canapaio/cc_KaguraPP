from cat.mad_hatter.decorators import tool, hook, plugin
# from cat.factory.llm import LLMSettings
from pydantic import BaseModel
from datetime import datetime, date
from cat.log import log
import os, re


@hook
def agent_prompt_prefix(prefix, cat):
    settings = cat.mad_hatter.get_plugin().load_settings()
    k_ppf = settings["kpp_path"] + settings["kpp_file"]
    if os.path.exists(k_ppf):
        with open(k_ppf, 'r') as f:
            prefix = f.read()
    else:
        prefix = settings["prompt_prefix"]
#    prefix += f"""
#<Conversazione>
#
#{re.sub(r'{}:/<>','_',cat.stringify_chat_history(latest_n=5))}
#
#</conversaione>
#"""    
    return prefix

@hook
def cat_recall_query(user_message, cat):
    prompt = f"""
# Genera un elenco di parole chiave in italiano e inglese relative al contenuto di 'testo-da-analizzare' per aiutare la ricercan nell'embedder seguendo le seguenti 'regole':
# <regole>
- Genera parole chiave in base al 'testo-da-analizzare' in italiano e iglese
- Elenca tutti i termini specifici dal 'testo-da-analizzare'
- Aggiungi chiavi non presenti che siano congrue con l'argomento
- NON COMMENTARE L'ELENCO
- Crea un elecon pulito privo di commenti
- Ignoras AI e Human
</regole>

# <testo-da-analizzare>

{re.sub(r'- AI','- KaguraAI',re.sub(r'- Human','- H',cat.stringify_chat_history(latest_n=10)))}

</testo-da-analizzare>
PS (crea solo un elenco di parole chiave in base alla sezione <testo-da-analizzare> usando le <regole> )

"""

    kpp_qwery = cat.llm(prompt)
    kpp_qwery += re.sub(r'AI|Human','_',cat.stringify_chat_history(latest_n=4))

#    cat.send_chat_message(LLMSettings("max_tokens"))

    return kpp_qwery

@hook
def agent_prompt_suffix(suffix, cat):
    settings = cat.mad_hatter.get_plugin().load_settings()
    suffix = """
<memory>
    <memory-past-conversations>
{episodic_memory}
    </memory-past-conversations>
    <memory-from-documents>
{declarative_memory}
    </memory-from-documents>
    <memory-from-executed-actions>
{tools_output}
    </memory-from-executed-actions>
</memory>
"""

    suffix += f"""
    Date Time:{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}
    ALWAYS answer in {settings['language']}
 """
    return suffix

@hook
def rabbithole_instantiates_splitter(text_splitter, cat):
    settings = cat.mad_hatter.get_plugin().load_settings()
    text_splitter._chunk_size = settings["chunk_size"]
    text_splitter._chunk_overlap = settings["chunk_overlap"]
    return text_splitter

@hook
def before_cat_recalls_episodic_memories(default_episodic_recall_config, cat):
    settings = cat.mad_hatter.get_plugin().load_settings()
    default_episodic_recall_config["k"] = settings["episodic_memory_k"]
    default_episodic_recall_config["threshold"] = settings["episodic_memory_threshold"]

    return default_episodic_recall_config


@hook
def before_cat_recalls_declarative_memories(default_declarative_recall_config, cat):
    settings = cat.mad_hatter.get_plugin().load_settings()
    default_declarative_recall_config["k"] = settings["declarative_memory_k"]
    default_declarative_recall_config["threshold"] = settings[
        "declarative_memory_threshold"
    ]

    return default_declarative_recall_config


@hook
def before_cat_recalls_procedural_memories(default_procedural_recall_config, cat):
    settings = cat.mad_hatter.get_plugin().load_settings()
    default_procedural_recall_config["k"] = settings["procedural_memory_k"]
    default_procedural_recall_config["threshold"] = settings[
        "procedural_memory_threshold"
    ]

    return default_procedural_recall_config
