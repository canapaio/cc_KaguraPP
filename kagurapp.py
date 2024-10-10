from cat.mad_hatter.decorators import tool, hook, plugin
from cat.factory.custom_llm import CustomOllama
#from cat.factory.llm import LLMSettings
from pydantic import BaseModel
from datetime import datetime, date
from cat.log import log
import os, re, copy

@hook
def cat_recall_query(user_message, cat):
    #settings = cat.mad_hatter.get_plugin().load_settings()
    settings = cat.mad_hatter.get_plugin().load_settings()
    #kpp_ctx_S = settiongs['']

    kprompt = f"""
Analizza la discussione contenuta in 'testo-da-analizzare' e genera una liste di parole chiavi congroue in italiano ed iglese seguendo le indicazioni contenute in 'regole':
<regole>
- Genera parole chiave in base al 'testo-da-analizzare' in italiano e inglese
- Elenca tutti i termini specifici dal 'testo-da-analizzare'
- Aggiungi chiavi non presenti che siano congrue con l'argomento
- NON COMMENTARE L'ELENCO
- Crea un elecon pulito privo di commenti
- Ignoras AI e Human
</regole>

<testo-da-analizzare>

{kre(cat.stringify_chat_history(latest_n=10))}

</testo-da-analizzare>
PS (crea solo un elenco di parole chiave in base alla sezione 'testo-da-analizzare' usando le 'regole' come indicato sopra )

"""

    #CustomOllama = (base_url='http://192.168.10.10:11434', model='qwen2.5-coder:latest', num_ctx=1024, repeat_last_n=64, repeat_penalty=1.1, temperature=0.8)
    log.info("======================================================")
    log.info(kprompt)

    llm_tmp = copy.deepcopy(cat._llm)

    alt_llm = cat.mad_hatter.get_plugin().load_settings().get('num_ctx', settings['kpp_ctx_S'])
    if alt_llm != '':
        llm_tmp.num_ctx = alt_llm
    alt_llm = cat.mad_hatter.get_plugin().load_settings().get('model', settings['kpp_model_s'])
    if alt_llm != '':
        llm_tmp.model = alt_llm
    #cat.send_chat_message(repr(llm_tmp))
    kpp_qwery = f"""
{kre(llm_tmp.invoke(kprompt).content)}
{kre(cat.stringify_chat_history(latest_n=4))}

"""
    #cat.send_chat_message
    log.info("======================================================")
    log.info(kpp_qwery)
    log.info("======================================================")

    return kpp_qwery


@hook
def agent_prompt_prefix(prefix: str, cat):
    settings = cat.mad_hatter.get_plugin().load_settings()
    k_ppf: str = settings["kpp_path"] + settings["kpp_file"]
    if os.path.exists(k_ppf):
        with open(k_ppf, 'r') as f:
            prefix = f.read()
    else:
        prefix = settings["prompt_prefix"]
    return prefix

@hook
def agent_prompt_suffix(suffix, cat):
    settings = cat.mad_hatter.get_plugin().load_settings()

#    suffix = f"""
#<Conversazione>
#    {kre(cat.stringify_chat_history(latest_n=5))}
#</conversaione>"""

    suffix = """
<oblio>
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
</oblio>
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

def kre(text: str):
    old: str
    new: str
    replacements = [
        ('- AI', '- KaguraAI'),
        ('- Human', '- H'),
        ('{', '/{'),
        ('}', '/}')
]
    for old, new in replacements:
        text = re.sub(old, new, text)
        
    return text