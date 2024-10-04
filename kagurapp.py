from cat.mad_hatter.decorators import tool, hook, plugin
from pydantic import BaseModel
from datetime import datetime, date
from cat.log import log
import os



@hook
def agent_prompt_prefix(prefix, cat):
    k_ppf = "./cat/plugins/cc_KaguraPP/promptprefix.txt"
#    k_ppf = settings["kpp_file"]
    if os.path.exists(k_ppf):
        with open(k_ppf, 'r') as f:
            prefix = f.read()
    else:
        settings = cat.mad_hatter.get_plugin().load_settings()
        prefix = settings["prompt_prefix"]
    return prefix

@hook
def cat_recall_query(user_message, cat):
    conversation_so_far  = cat.stringify_chat_history(latest_n=5)

    prompt = f"""
Genera un elenco in markdown di parole chiave congrue da <testo-da-analizzare> per agevolare l'embedder a trovare gli argomenti memorizzati nel sistema:
Elenca tutti i termini specifici 
Aggiungi eventuali chiavi non presenti che siano congrue con l'argomento

NON COMMENTARE L'ELENCO
Esempio:
- Parola 1
- Parola 2
...

<testo-da-analizzare>

{conversation_so_far}

</testo-da-analizzare>

PS (NON COMMENTARE L'ELENCO)

"""
    
    compressed_query = cat.llm(prompt)
    compressed_query += cat.stringify_chat_history(latest_n=3)

    return compressed_query

@hook
def agent_prompt_suffix(suffix, cat):
    settings = cat.mad_hatter.get_plugin().load_settings()
#    cat.log 
    suffix = f"""
<Conversazione>
{cat.stringify_chat_history(latest_n=5)}
</conversaione>

"""

    suffix += """
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
