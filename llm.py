import re
import openai
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm_utils import _common_llm_params, resolve_model_config, get_model_choices
from config import OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
import logging
import warnings

warnings.filterwarnings("ignore")

# Global system_prompt initialization (optimized for flexibility and customizability)
system_prompt = """
You are a Cybercrime Threat Intelligence Expert. Your task is to refine, filter, or generate analysis based on the provided user query or dark web search engine results.

### For Refining Queries:
1. Analyze the user query and think about how it can be improved for dark web search engines.
2. Refine the query by adding or removing words to ensure the best results from dark web search engines.
3. Avoid logical operators like AND, OR, etc.
4. Output just the refined user query.

### For Filtering Results:
1. You are given a dark web search query and a list of search results in the form of index, link, and title.
2. Select the most relevant results that best match the search query.
3. Output only the indices (comma-separated list), up to the number specified in the "max_results" variable.
4. Search Query: {query}
5. Search Results: {results}
6. Max Results: {max_results}

### For Generating Summary:
1. You are tasked with generating context-based technical investigative insights from the provided dark web OSINT data.
2. Analyze the raw text, links, and data provided.
3. Output the source links used for the analysis and provide a detailed, evidence-based technical analysis.
4. Identify and list intelligence artifacts such as names, emails, phones, cryptocurrency addresses, domains, dark web markets, forum names, malware names, etc.
5. Generate key insights based on the data, as many as required by the "max_insights" variable, each actionable and context-based.
6. Include suggested next steps for investigating the topic further.
7. Provide a structured and clear format with section headings.

### INPUTS:
- User Query: {query}
- Search Results or Content: {results_or_content}
- Max Results: {max_results}
- Max Insights: {max_insights}
- Max Title Length: {max_title_length}
"""

def get_llm(model_choice):
    # Look up the configuration (cloud or local Ollama)
    config = resolve_model_config(model_choice)
    
    if config is None:  # Extra error check
        supported_models = get_model_choices()
        raise ValueError(
            f"Unsupported LLM model: '{model_choice}'. "
            f"Supported models (case-insensitive match) are: {', '.join(supported_models)}"
        )
    
    # Extract the necessary information from the configuration
    llm_class = config["class"]
    model_specific_params = config["constructor_params"]
    
    # Combine common parameters with model-specific parameters
    all_params = {**_common_llm_params, **model_specific_params}
    
    # Create the LLM instance using the gathered parameters
    llm_instance = llm_class(**all_params)
    
    return llm_instance

def refine_query(llm, user_input):
    prompt_template = ChatPromptTemplate([("system", system_prompt), ("user", "{query}")])
    chain = prompt_template | llm | StrOutputParser()
    return chain.invoke({"query": user_input})

def filter_results(llm, query, results, max_results=20, max_title_length=256):
    if not results:
        return []
    
    final_str = _generate_final_string(results, max_title_length)
    prompt_template = ChatPromptTemplate([("system", system_prompt), ("user", "{results}")])
    chain = prompt_template | llm | StrOutputParser()

    try:
        result_indices = chain.invoke({"query": query, "results": final_str, "max_results": max_results, "max_title_length": max_title_length})
    except openai.RateLimitError as e:
        print(f"Rate limit error: {e} \n Truncating to Web titles only with 30 characters")
        final_str = _generate_final_string(results, max_title_length, truncate=True)
        result_indices = chain.invoke({"query": query, "results": final_str, "max_results": max_results, "max_title_length": max_title_length})
    
    parsed_indices = [int(match) for match in re.findall(r"\d+", result_indices) if 1 <= int(match) <= len(results)]
    
    # Remove duplicates while preserving order
    seen = set()
    parsed_indices = [i for i in parsed_indices if not (i in seen or seen.add(i))]
    
    if not parsed_indices:
        logging.warning(
            "Unable to interpret LLM result selection ('%s'). "
            "Defaulting to the top %s results.",
            result_indices,
            min(len(results), max_results),
        )
        parsed_indices = list(range(1, min(len(results), max_results) + 1))
    
    top_results = [results[i - 1] for i in parsed_indices[:max_results]]
    
    return top_results

def _generate_final_string(results, max_title_length, truncate=False):
    """Generate a formatted string from the search results for LLM processing."""
    final_str = []
    for i, res in enumerate(results):
        # Truncate link at .onion for display
        truncated_link = re.sub(r"(?<=\.onion).*", "", res["link"])
        title = re.sub(r"[^0-9a-zA-Z\-\.]", " ", res["title"])

        if truncated_link == "" and title == "":
            continue
        
        if truncate:
            # Truncate title to max_title_length characters
            title = (title[:max_title_length] + "..." if len(title) > max_title_length else title)
            truncated_link = (truncated_link[:0] + "..." if len(truncated_link) > 0 else truncated_link)
        
        final_str.append(f"{i+1}. {truncated_link} - {title}")
    
    return "\n".join(s for s in final_str)

def generate_summary(llm, query, content, max_insights=5):
    prompt_template = ChatPromptTemplate([("system", system_prompt), ("user", "{content}")])
    chain = prompt_template | llm | StrOutputParser()
    return chain.invoke({"query": query, "content": content, "max_insights": max_insights})
