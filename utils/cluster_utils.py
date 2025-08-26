from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import field
import json
import logging
import math
import numbers
import random
import re
import numpy as np
import tiktoken
import umap
import copy
import asyncio
from abc import ABC, abstractmethod
from typing import Any, List, Optional
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import pdist, squareform
import warnings
from tqdm import tqdm
from collections import Counter, defaultdict
from itertools import combinations
from utils.text_processing_utils import split_string_by_multi_markers, clean_str, is_float_regex
from prompt import GRAPH_FIELD_SEP, PROMPTS
from utils.file_and_instance_utils import write_jsonl, write_jsonl_force
# Setup simple logging
logger = logging.getLogger(__name__)
ENCODER = None

def check_test(entities):
    e_l=[]
    max_len=len(entities)
    for layer in entities:
        temp_e=[]
        if type(layer) != list:
            temp_e.append(layer['entity_name'])
            e_l.append(temp_e)
            continue
        for item in layer:
            temp_e.append(item['entity_name'])
        e_l.append(temp_e)
        
    for index,layer in enumerate(entities):
        if type(layer) != list or index==max_len-1:
            break
        for item in layer:
            if item['parent'] not in e_l[index+1]:
                print(item['entity_name'],item['parent'])
def extract_first_complete_json(s: str):
    """Extract the first complete JSON object from the string using a stack to track braces."""
    stack = []
    first_json_start = None
    
    for i, char in enumerate(s):
        if char == '{':
            stack.append(i)
            if first_json_start is None:
                first_json_start = i
        elif char == '}':
            if stack:
                start = stack.pop()
                if not stack:
                    first_json_str = s[first_json_start:i+1]
                    try:
                        # Attempt to parse the JSON string
                        return json.loads(first_json_str.replace("\n", ""))
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decoding failed: {e}. Attempted string: {first_json_str[:50]}...")
                        return None
                    finally:
                        first_json_start = None
    logger.warning("No complete JSON object found in the input string.")
    return None
def extract_json_from_cluster(s:str):
    import re
    s=s.replace('*', '')
    entity_name = re.search(r"Aggregate Entity Name:\s*(.+)", s).group(1).strip()
    entity_description = re.search(
        r"Aggregate Entity Description:\s*(.+?)\n\nFindings:", s, re.DOTALL
    ).group(1).strip()

    # 提取 findings
    pattern = r"<summary_(\d+)>:\s*(.*?)\s*<explanation_\1>:\s*(.*?)(?=\n<summary_\d+>:|\Z)"
    matches = re.findall(pattern, s, re.DOTALL)

    findings = []
    for _, summary, explanation in matches:
        findings.append({
            "summary": summary.strip().replace('\n', ' '),
            "explanation": explanation.strip().replace('\n', ' ')
        })

    # 构造最终 JSON
    result = {
        "entity_name": entity_name,
        "entity_description": entity_description,
        "findings": findings
    }

    # 输出 JSON 字符串（可选择写入文件）
    return result
def parse_value(value: str):
    """Convert a string value to its appropriate type (int, float, bool, None, or keep as string). Work as a more broad 'eval()'"""
    value = value.strip()

    if value == "null":
        return None
    elif value == "true":
        return True
    elif value == "false":
        return False
    else:
        # Try to convert to int or float
        try:
            if '.' in value:  # If there's a dot, it might be a float
                return float(value)
            else:
                return int(value)
        except ValueError:
            # If conversion fails, return the value as-is (likely a string)
            return value.strip('"')  # Remove surrounding quotes if they exist
def extract_values_from_json(json_string, keys=["reasoning", "answer", "data"], allow_no_quotes=False):
    """Extract key values from a non-standard or malformed JSON string, handling nested objects."""
    extracted_values = {}
    
    # Enhanced pattern to match both quoted and unquoted values, as well as nested objects
    regex_pattern = r'(?P<key>"?\w+"?)\s*:\s*(?P<value>{[^}]*}|".*?"|[^,}]+)'
    
    for match in re.finditer(regex_pattern, json_string, re.DOTALL):
        key = match.group('key').strip('"')  # Strip quotes from key
        value = match.group('value').strip()

        # If the value is another nested JSON (starts with '{' and ends with '}'), recursively parse it
        if value.startswith('{') and value.endswith('}'):
            extracted_values[key] = extract_values_from_json(value)
        else:
            # Parse the value into the appropriate type (int, float, bool, etc.)
            extracted_values[key] = parse_value(value)

    if not extracted_values:
        logger.warning("No values could be extracted from the string.")
    
    return extracted_values


def save_failed_json_response(response: str, prompt: str = None, error_msg: str = None):
    """Log failed JSON response for debugging (no file output)."""
    try:
        logger.error("Failed JSON response detected:")
        if error_msg:
            logger.error(f"Error: {error_msg}")
        if prompt:
            logger.error(f"Prompt (first 200 chars): {prompt[:200]}...")
        logger.error(f"Response (first 500 chars): {response[:500]}...")
        
    except Exception as save_error:
        logger.error(f"Failed to log JSON debug info: {save_error}")

def convert_response_to_json(response: str, prompt: str = None) -> dict:
    """Convert response string to JSON, with comprehensive error logging."""
    prediction_json = extract_first_complete_json(response)
    
    if prediction_json is None:
        logger.info("Attempting to extract values from a non-standard JSON string...")
        prediction_json = extract_values_from_json(response, allow_no_quotes=True)
        
        if not prediction_json:
            # Save the failed response for debugging
            error_msg = "Both JSON extraction methods failed"
            logger.error(f"{error_msg}. Response length: {len(response)} chars")
            save_failed_json_response(response, prompt, error_msg)
    
    if not prediction_json:
        logger.error("Unable to extract meaningful data from the response.")
        return {}
    else:
        logger.info("JSON data successfully extracted.")
    
    return prediction_json

def save_duplicate_analysis_report(duplicate_analysis: dict, layer: int = 0):
    """Log duplicate analysis summary (no file output)."""
    if not duplicate_analysis["has_duplicates"]:
        return
    
    try:
        num_duplicates = len(duplicate_analysis.get("duplicate_pairs", []))
        logger.warning(f"Layer {layer}: Found {num_duplicates} duplicate entity pairs")
        
        # Log a few examples
        for i, pair in enumerate(duplicate_analysis.get("duplicate_pairs", [])[:3]):  # Show first 3
            if "entity1" in pair and "entity2" in pair:
                logger.warning(f"  Duplicate {i+1}: '{pair['entity1']['name']}' <-> '{pair['entity2']['name']}' (distance: {pair['distance']:.3f})")
        
        if num_duplicates > 3:
            logger.warning(f"  ... and {num_duplicates - 3} more duplicates")
        
    except Exception as e:
        logger.error(f"Failed to log duplicate analysis: {e}")

def generate_clustering_quality_report(layer: int, nodes: list, embeddings: np.ndarray, clusters: list, duplicate_analysis: dict = None):
    """Generate clustering quality report (log only, no file output)."""
    try:
        # Calculate clustering statistics
        cluster_sizes = {}
        for i, cluster in enumerate(clusters):
            for cluster_id in cluster:
                cluster_id = int(cluster_id)  # Convert numpy int64 to Python int
                if cluster_id not in cluster_sizes:
                    cluster_sizes[cluster_id] = 0
                cluster_sizes[cluster_id] += 1
        
        # Create quality indicators
        quality_indicators = {
            "entities_per_cluster": float(len(nodes) / len(cluster_sizes)) if cluster_sizes else 0.0,
            "duplicate_rate": float(duplicate_analysis.get("duplicate_rate", 0.0)) if duplicate_analysis else 0.0,
            "singleton_clusters": int(sum(1 for size in cluster_sizes.values() if size == 1)),
            "large_clusters": int(sum(1 for size in cluster_sizes.values() if size > 50))
        }
        
        # Log key quality indicators
        logger.info(f"Layer {layer} Quality Summary:")
        logger.info(f"  Total entities: {len(nodes)}")
        logger.info(f"  Total clusters: {len(cluster_sizes)}")
        logger.info(f"  Avg entities per cluster: {quality_indicators['entities_per_cluster']:.1f}")
        logger.info(f"  Duplicate rate: {quality_indicators['duplicate_rate']:.1f}%")
        logger.info(f"  Singleton clusters: {quality_indicators['singleton_clusters']}")
        
        if quality_indicators['duplicate_rate'] > 20.0:
            logger.warning(f"High duplicate rate ({quality_indicators['duplicate_rate']:.1f}%) may indicate data quality issues")
        
        if quality_indicators['singleton_clusters'] > len(cluster_sizes) * 0.5:
            logger.warning(f"Many singleton clusters ({quality_indicators['singleton_clusters']}) may indicate poor clustering")
        
        return quality_indicators
        
    except Exception as e:
        logger.error(f"Failed to generate clustering quality report: {e}")
        return None
def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    tokens = ENCODER.encode(content)
    return tokens

# Set a random seed for reproducibility
RANDOM_SEED = 224
random.seed(RANDOM_SEED)
def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int):
    """Truncate a list of data by token size"""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string_by_tiktoken(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data

def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: int = 15,
    metric: str = "cosine",
) -> np.ndarray:
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    reduced_embeddings = umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings


def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    reduced_embeddings = umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings


def analyze_embedding_duplicates(embeddings: np.ndarray, entities: list = None, threshold: float = 1e-10) -> dict:
    """
    Analyze duplicate embeddings and return detailed information.
    
    Args:
        embeddings: Array of embeddings
        entities: List of entity dictionaries (optional, for detailed reporting)
        threshold: Distance threshold for considering embeddings as duplicates
        
    Returns:
        Dictionary with duplicate analysis results
    """
    if len(embeddings) <= 1:
        return {"has_duplicates": False, "duplicate_pairs": [], "duplicate_rate": 0.0}
    
    # Compute pairwise distances
    distances = pdist(embeddings)
    distance_matrix = squareform(distances)
    
    # Find duplicate pairs
    duplicate_indices = np.where((distance_matrix < threshold) & (distance_matrix > 0))
    duplicate_pairs = []
    seen_pairs = set()
    
    for i, j in zip(duplicate_indices[0], duplicate_indices[1]):
        if i < j:  # Avoid duplicate pairs (i,j) and (j,i)
            pair_key = (i, j)
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                
                pair_info = {
                    "indices": (i, j),
                    "distance": distance_matrix[i, j],
                }
                
                # Add entity information if available
                if entities is not None and len(entities) > max(i, j):
                    pair_info["entity1"] = {
                        "name": entities[i].get("entity_name", f"Entity_{i}"),
                        "type": entities[i].get("entity_type", "unknown"),
                        "description": entities[i].get("description", "")[:100] + "..." if len(entities[i].get("description", "")) > 100 else entities[i].get("description", ""),
                        "source": entities[i].get("source_id", "unknown")
                    }
                    pair_info["entity2"] = {
                        "name": entities[j].get("entity_name", f"Entity_{j}"),
                        "type": entities[j].get("entity_type", "unknown"), 
                        "description": entities[j].get("description", "")[:100] + "..." if len(entities[j].get("description", "")) > 100 else entities[j].get("description", ""),
                        "source": entities[j].get("source_id", "unknown")
                    }
                
                duplicate_pairs.append(pair_info)
    
    duplicate_rate = len(duplicate_pairs) / len(embeddings) * 100
    
    return {
        "has_duplicates": len(duplicate_pairs) > 0,
        "duplicate_pairs": duplicate_pairs,
        "duplicate_rate": duplicate_rate,
        "total_entities": len(embeddings),
        "duplicate_count": len(duplicate_pairs)
    }

def clean_embeddings(embeddings: np.ndarray, entities: list = None, min_variance: float = 1e-6) -> tuple:
    """
    Clean embeddings by handling NaN values, duplicates, and numerical issues.
    
    Returns:
        Tuple of (cleaned_embeddings, duplicate_analysis) if entities provided,
        otherwise just cleaned_embeddings for backward compatibility
    """
    # Check for NaN or infinite values
    valid_mask = np.isfinite(embeddings).all(axis=1)
    if not valid_mask.all():
        invalid_count = (~valid_mask).sum()
        logger.warning(f"Found {invalid_count} embeddings with NaN/inf values, removing them")
        embeddings = embeddings[valid_mask]
        if entities is not None:
            entities = [entities[i] for i in range(len(entities)) if valid_mask[i]]
    
    if len(embeddings) == 0:
        raise ValueError("No valid embeddings after cleaning")
    
    # Analyze duplicates if entities are provided
    duplicate_analysis = None
    if entities is not None:
        duplicate_analysis = analyze_embedding_duplicates(embeddings, entities, threshold=1e-10)
        
        # Add small random noise to handle duplicate embeddings
        if duplicate_analysis["has_duplicates"]:
            logger.warning(f"Found {duplicate_analysis['duplicate_count']} duplicate/near-duplicate embedding pairs ({duplicate_analysis['duplicate_rate']:.1f}% rate)")
            
            # Log first few duplicate pairs for debugging
            for i, pair in enumerate(duplicate_analysis["duplicate_pairs"][:3]):  # Show first 3 pairs
                if "entity1" in pair and "entity2" in pair:
                    logger.info(f"  Duplicate {i+1}: '{pair['entity1']['name']}' ({pair['entity1']['type']}) <-> '{pair['entity2']['name']}' ({pair['entity2']['type']}) [distance: {pair['distance']:.2e}]")
            
            if len(duplicate_analysis["duplicate_pairs"]) > 3:
                logger.info(f"  ... and {len(duplicate_analysis['duplicate_pairs']) - 3} more duplicate pairs")
            
            noise_scale = min_variance * np.sqrt(embeddings.shape[1])
            noise = np.random.normal(0, noise_scale, embeddings.shape)
            embeddings = embeddings + noise
    else:
        # Backward compatibility - simple duplicate detection without entity details
        if len(embeddings) > 1:
            distances = pdist(embeddings)
            if np.any(distances < 1e-10):
                logger.warning("Found duplicate/near-duplicate embeddings, adding noise")
                noise_scale = min_variance * np.sqrt(embeddings.shape[1])
                noise = np.random.normal(0, noise_scale, embeddings.shape)
                embeddings = embeddings + noise
    
    # Normalize to prevent overflow/underflow
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Return based on whether we have analysis results
    if duplicate_analysis is not None:
        return embeddings, duplicate_analysis
    else:
        return embeddings

def fit_gaussian_mixture(n_components, embeddings, random_state):
    # Clean embeddings before fitting
    embeddings_clean = clean_embeddings(embeddings)
    
    # Ensure n_components doesn't exceed number of samples
    n_components = min(n_components, len(embeddings_clean))
    
    if n_components < 1:
        n_components = 1
    
    gm = GaussianMixture(
        n_components=n_components,
        random_state=random_state,
        n_init=5,
        init_params='k-means++',
        reg_covar=1e-6  # Add regularization to prevent singular covariance matrices
        )
    
    # Suppress warnings during fit
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        gm.fit(embeddings_clean)
    
    return gm.bic(embeddings_clean)


def get_optimal_clusters(embeddings, max_clusters=50, random_state=0, rel_tol=1e-3):
    max_clusters = min(len(embeddings), max_clusters)
    n_clusters = np.arange(1, max_clusters)
    bics = []
    prev_bic = float('inf')
    for n in tqdm(n_clusters):
        bic = fit_gaussian_mixture(n, embeddings, random_state)
        # print(bic)
        bics.append(bic)
        # early stop - add proper handling for edge cases
        if prev_bic != float('inf') and abs(prev_bic) > 1e-10:  # Avoid division by zero/inf
            relative_change = abs(prev_bic - bic) / abs(prev_bic)
            if relative_change < rel_tol:
                break
        prev_bic = bic
    optimal_clusters = n_clusters[np.argmin(bics)]
    return optimal_clusters


def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0,cluster_size: int = 20):
    # Clean embeddings first
    embeddings_clean = clean_embeddings(embeddings)
    
    n_clusters = max(len(embeddings_clean) // cluster_size, get_optimal_clusters(embeddings_clean))
    n_clusters = min(n_clusters, len(embeddings_clean))  # Ensure n_clusters <= n_samples
    
    if n_clusters < 1:
        n_clusters = 1
    
    gm = GaussianMixture(
            n_components=n_clusters, 
            random_state=random_state, 
            n_init=5,
            init_params='k-means++',
            reg_covar=1e-6  # Add regularization
            )
    
    # Suppress warnings during fit
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        gm.fit(embeddings_clean)
        probs = gm.predict_proba(embeddings_clean)        # [num, cluster_num]
    
    # labels = [np.where(prob > threshold)[0] for prob in probs]
    labels = [[np.argmax(prob)] for prob in probs]
    return labels, n_clusters


def perform_clustering(
    embeddings: np.ndarray, dim: int, threshold: float, verbose: bool = False, cluster_size: int = 20, entities: list = None, layer: int = 0
) -> tuple:
    # Analyze duplicates and get detailed analysis
    duplicate_analysis = None
    if entities is not None and len(entities) == len(embeddings):
        _, duplicate_analysis = clean_embeddings(embeddings, entities)
        if duplicate_analysis["duplicate_rate"] > 10.0:  # More than 10% duplicates
            logger.warning(f"High duplicate rate detected: {duplicate_analysis['duplicate_rate']:.1f}% - this may indicate data quality issues")
    
    reduced_embeddings_global = global_cluster_embeddings(embeddings, min(dim, len(embeddings) -2))
    global_clusters, n_global_clusters = GMM_cluster(     # (num, 2)
        reduced_embeddings_global, threshold,cluster_size=cluster_size
    )
    
    # Generate quality report if entities are provided
    if entities is not None:
        generate_clustering_quality_report(layer, entities, embeddings, global_clusters, duplicate_analysis)
    
    # Debug check
    if len(global_clusters) != len(embeddings):
        logger.warning(f"Cluster assignment mismatch: {len(global_clusters)} assignments for {len(embeddings)} embeddings")
    
    return global_clusters, duplicate_analysis

    # all_clusters = [[] for _ in range(len(embeddings))]
    # embedding_to_index = {tuple(embedding): idx for idx, embedding in enumerate(embeddings)}
    # for i in tqdm(range(n_global_clusters)):
    #     global_cluster_embeddings_ = embeddings[
    #         np.array([i in gc for gc in global_clusters])
    #     ]  #找到当前簇的embedding
    #     if verbose:
    #         logging.info(
    #             f"Nodes in Global Cluster {i}: {len(global_cluster_embeddings_)}"
    #         )
    #     if len(global_cluster_embeddings_) == 0:
    #         continue

    #     # embedding indices #反向取idx
    #     indices = [
    #         embedding_to_index[tuple(embedding)]
    #         for embedding in global_cluster_embeddings_
    #     ]

    #     # update
    #     for idx in indices:
    #         all_clusters[idx].append(i)

    # all_clusters = [np.array(cluster) for cluster in all_clusters]

    # if verbose:
    #     logging.info(f"Total Clusters: {len(n_global_clusters)}")
    return global_clusters

def enclose_string_with_quotes(content: Any) -> str:
    """Enclose a string with quotes"""
    if isinstance(content, numbers.Number):
        return str(content)
    content = str(content)
    content = content.strip().strip("'").strip('"')
    return f'"{content}"'
def list_of_list_to_csv(data: list[list]):
    return "\n".join(
        [
            ",\t".join([f"{enclose_string_with_quotes(data_dd)}" for data_dd in data_d])
            for data_d in data
        ]
    )
def get_direct_relations(set1,set2,relations):
    results={k:v for k,v in relations.items() if (k[0]in set1 and k[1] in set2) or (k[0] in set2 and k[1] in set1)}
    return results
    
async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        source_id=edge_source_id,
    )


class ClusteringAlgorithm(ABC):
    @abstractmethod
    def perform_clustering(self, embeddings: np.ndarray, **kwargs) -> List[List[int]]:
        pass
def _pack_single_community_describe(
    entitys,
    relations,
    max_token_size: int = 12000,
    global_config: dict = {},
) -> str:
   
    node_fields = ["id", "entity", "type", "description", "degree"]
    edge_fields = ["id", "source", "target", "description", "rank"]
    nodes_list_data = [
        [
            i,
            entity.get("entity_name"),
            entity.get("entity_type", "UNKNOWN"),
            entity.get("description", "UNKNOWN"),
            entity.get("degree",1)
        ]
        for i,entity in enumerate(entitys)
    ]
    nodes_list_data = sorted(nodes_list_data, key=lambda x: x[-1], reverse=True)
    nodes_may_truncate_list_data = truncate_list_by_token_size(
        nodes_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )
    edges_list_data = [
        [
            i,
            relation.get("src_tgt"),
            relation.get("tgt_src"),
            relation.get("description", "UNKNOWN"),
        ]
        for i, relation in enumerate(relations.values())
    ]
    edges_list_data = sorted(edges_list_data, key=lambda x: x[-1], reverse=True)
    edges_may_truncate_list_data = truncate_list_by_token_size(
        edges_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )

    truncated = len(nodes_list_data) > len(nodes_may_truncate_list_data) or len(
        edges_list_data
    ) > len(edges_may_truncate_list_data)

  
    nodes_describe = list_of_list_to_csv([node_fields] + nodes_may_truncate_list_data)
    edges_describe = list_of_list_to_csv([edge_fields] + edges_may_truncate_list_data)
    return f"""
-----Entities-----
```csv
{nodes_describe}
```
-----Relationships-----
```csv
{edges_describe}
```"""
def process_cluster( 
    use_llm_func, embeddings_func, \
    clusters,label,nodes,community_report_prompt,\
        relations,generate_relations,layer,temp_clusters_nodes
):
    indices = [i for i, cluster in enumerate(clusters) if label in cluster]
                # Add the corresponding nodes to the node_clusters list
    cluster_nodes = [nodes[i] for i in indices]
    
    # Base case: if the cluster only has one node, do not attempt to recluster it
    logging.info(f"[Label{str(int(label))} Size: {len(cluster_nodes)}]")
    if len(cluster_nodes) == 1:
        cluster_nodes[0]['parent']=cluster_nodes[0]['entity_name']
        return {
        'community_data': None,
        'temp_node': cluster_nodes[0],
        'index':indices
        }
    name_set=[node['entity_name'] for node in cluster_nodes]
    cluster_intern_relation={**get_direct_relations(name_set,name_set,relations),
        **get_direct_relations(name_set,name_set,generate_relations)}
    describe=_pack_single_community_describe(cluster_nodes,cluster_intern_relation)
    hint_prompt=community_report_prompt.format(input_text=describe)
    response = use_llm_func(hint_prompt)
    data = convert_response_to_json(response, hint_prompt)
    
    # Validate required fields and provide fallbacks
    if not data or not isinstance(data, dict):
        logger.warning(f"Invalid or empty response data for cluster {label}. Using fallback.")
        data = {}
    
    # Ensure required fields exist with fallbacks
    entity_name = data.get('entity_name') or f"Cluster_{int(label)}_Layer_{layer}"
    entity_description = data.get('entity_description') or f"Aggregate entity representing {len(cluster_nodes)} related entities"
    
    # Validate entity_name is not empty or just whitespace
    if not entity_name.strip():
        entity_name = f"Cluster_{int(label)}_Layer_{layer}"
        
    # Validate entity_description is not empty or just whitespace
    if not entity_description.strip():
        entity_description = f"Aggregate entity representing {len(cluster_nodes)} related entities"
    
    data['level'] = layer
    data['children'] = [n['entity_name'] for n in cluster_nodes]
    data['source_id'] = "|".join(set([n['source_id'] for n in cluster_nodes]))
    data['entity_name'] = entity_name
    data['entity_description'] = entity_description

    temp_node = {
        'entity_name': entity_name,
        'description': entity_description,
        'source_id': data['source_id'],
        'entity_type': "aggregate entity",
        'degree': 1,
        'vector': embeddings_func(entity_description),
    }


    return {
        'community_data': data,
        'temp_node': temp_node,
        'index':indices
    }
def process_relation( 
    use_llm_func,community_report,maybe_edge,relations,generate_relations,\
     cluster_cluster_relation_prompt,layer,tokenizer,max_depth
    
):
    cluster1_nodes=community_report[maybe_edge[0]]['children']
    cluster2_nodes=community_report[maybe_edge[1]]['children']
    
    threshold=min(len(cluster1_nodes)*0.2,len(cluster2_nodes)*0.2)
    # threshold=1
    exists_relation={**get_direct_relations(cluster1_nodes,cluster2_nodes,relations),
    **get_direct_relations(cluster1_nodes,cluster2_nodes,generate_relations)}
    if exists_relation=={}:
        return None
            
    cluster1_description=community_report[maybe_edge[0]]['findings']
    cluster2_description=community_report[maybe_edge[1]]['findings']
    relation_infromation=[ 
                                f"relationship<|>{v['src_tgt']}<|>{v['tgt_src']}<|>{v['description']} "
                                for k,v in exists_relation.items()
                                ]
    temp_relations={}
    tokens=len(tokenizer.encode("\n".join(relation_infromation)))
    gene_tokens=(layer+1)*40
    allowed_tokens=(max_depth-layer)*40*2
    # allowed_tokens=100000
    # allowed_tokens=
    if tokens>allowed_tokens:
        print(f"{tokens} > {allowed_tokens}，allowed tokens\n{maybe_edge[0]} & {maybe_edge[1]} in processing")
        exact_prompt=cluster_cluster_relation_prompt.format(entity_a=maybe_edge[0],entity_b=maybe_edge[1],\
            entity_a_description=cluster1_description,entity_b_description=cluster2_description,\
                relation_information="\n".join(relation_infromation),tokens=gene_tokens)
        
        response = use_llm_func(exact_prompt)
        temp_relations[maybe_edge]={
                            'src_tgt':maybe_edge[0],
                            'tgt_src':maybe_edge[1],
                            'description':response,
                            'weight':1,
                            'level':layer+1
                        }
    else:
        print(f"{tokens} < {allowed_tokens} allowed tokens")
        temp_relations[maybe_edge]={
                            'src_tgt':maybe_edge[0],
                            'tgt_src':maybe_edge[1],
                            'description':"\n".join(relation_infromation),
                            'weight':1,
                            'level':layer+1
                        }
    return temp_relations

class Hierarchical_Clustering(ClusteringAlgorithm):
    def perform_clustering(
        self,
        global_config: dict,
        entities: dict,
        relations:dict,
        max_length_in_cluster: int = 60000,
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        reduction_dimension: int = 2,
        cluster_threshold: float = 0.1,
        verbose: bool = False,
        threshold: float = 0.98, # 0.99
        thredshold_change_rate: float = 0.05,
        WORKING_DIR: str = None,
        max_workers: int =8,
        cluster_size: int=20,
    ) -> List[dict]:
        use_llm_func: callable = global_config["use_llm_func"]
        embeddings_func: callable = global_config["embeddings_func"]
        # Get the embeddings from the nodes
        nodes = list(entities.values())
        embeddings = np.array([x["vector"] for x in nodes])
        generate_relations={}
        max_workers=global_config['max_workers']
        community_report={}
        all_nodes=[]
        all_nodes.append(nodes)
        community_report_prompt = PROMPTS["aggregate_entities"]
        cluster_cluster_relation_prompt = PROMPTS["cluster_cluster_relation"]
        max_depth=round(math.log(len(nodes),cluster_size))+1
        for layer in range(max_depth):
            logging.info(f"############ Layer[{layer}] Clustering ############")
            # Perform the clustering
            if  len(nodes) <= 2:
                print("当前簇数小于2，停止聚类")
                break
            clusters, duplicate_analysis = perform_clustering(
                embeddings, dim=reduction_dimension, threshold=cluster_threshold, cluster_size=cluster_size, entities=nodes, layer=layer
            )
            temp_clusters_nodes = []
            # Initialize an empty list to store the clusters of nodes
            # Iterate over each unique label in the clusters
            unique_clusters = np.unique(np.concatenate(clusters))
            logging.info(f"[Clustered Label Num: {len(unique_clusters)} / Last Layer Total Entity Num: {len(nodes)}]")
            # calculate the number of nodes belong to each cluster
            # cluster_sizes = Counter(np.concatenate(clusters))
            # # calculate cluster sparsity
            # cluster_sparsity = 1 - sum([x * (x - 1) for x in cluster_sizes.values()])/(len(nodes) * (len(nodes) - 1))
            # cluster_sparsity_change_rate = (abs(cluster_sparsity - pre_cluster_sparsity) / pre_cluster_sparsity)
            # pre_cluster_sparsity = cluster_sparsity
            # logging.info(f"[Cluster Sparsity: {round(cluster_sparsity, 4) * 100}%]")
            
            # if cluster_sparsity_change_rate <= thredshold_change_rate:
            #     logging.info(f"[Stop Clustering at Layer{layer} with Cluster Sparsity Change Rate {round(cluster_sparsity_change_rate, 4) * 100}%]")
            #     break
            # summarize
            if len(unique_clusters) <=4:
                print(f"Current # clusters < 5，Stop clustering")
                break
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        process_cluster, 
                        use_llm_func, embeddings_func, clusters, label, nodes,
                        community_report_prompt, relations, generate_relations, layer,temp_clusters_nodes
                    )
                    for label in unique_clusters
                ]
                for future in tqdm(as_completed(futures), total=len(futures)):
                    result = future.result()
                    temp_clusters_nodes.append(result['temp_node'])    
                    for index in result['index']:
                        nodes[index]['parent']=result['temp_node']['entity_name']
                    if result['community_data'] is not None:    
                        title=result['community_data']['entity_name']
                        community_report[title] = result['community_data']
                    
           
            

            temp_cluster_relation=[i['entity_name'] for i in temp_clusters_nodes if i['entity_name'] in community_report.keys()] 
            temp_relations={}     
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        process_relation, use_llm_func,community_report,maybe_edge,\
                            relations,generate_relations, cluster_cluster_relation_prompt,layer,tokenizer,max_depth
                    )
                    for maybe_edge in list(combinations(temp_cluster_relation,2))
                ]

                for future in tqdm(as_completed(futures), total=len(futures)):
                    result = future.result()
                    if result!=None:
                        for k,v in result.items():
                            temp_relations[k]=v
            for k,v in temp_relations.items():
                generate_relations[k]=v
                   
             
            
               
            # update nodes to be clustered in the next layer
            nodes = copy.deepcopy([x for x in temp_clusters_nodes if "entity_name" in x.keys()])
            # filter the duplicate entities
            seen = set()        
            unique_nodes = []
            for item in nodes:
                entity_name = item['entity_name']
                if entity_name not in seen:
                    seen.add(entity_name)
                    unique_nodes.append(item)
            nodes = unique_nodes
            for index,i in enumerate(unique_nodes): #再进行embedding时发现，有个元素的vector不是np而是list
                vec=i["vector"]
                if type(vec)==list:
                    # Convert list to numpy array and ensure 2D shape
                    vec_array = np.array(vec)
                    if vec_array.ndim == 1:
                        vec_array = vec_array.reshape(1, -1)
                    unique_nodes[index]["vector"] = vec_array
                    print(f"Converted list to array for index {index}, shape: {vec_array.shape}")
                elif vec.ndim == 1:
                    # Ensure 2D shape for 1D arrays
                    unique_nodes[index]["vector"] = vec.reshape(1, -1)
                    print(f"Reshaped 1D array for index {index}, shape: {vec.shape} -> {unique_nodes[index]['vector'].shape}")
            
            # Ensure all vectors have consistent shape for embedding array creation
            vectors = []
            expected_dim = None
            
            for i, x in enumerate(unique_nodes):
                vec = x["vector"]
                if isinstance(vec, list):
                    vec = np.array(vec)
                
                # Flatten any vector to 1D first
                vec_flat = vec.flatten()
                
                # Set expected dimension from first vector
                if expected_dim is None:
                    expected_dim = len(vec_flat)
                    print(f"Setting expected embedding dimension to {expected_dim}")
                
                # Ensure all vectors have the same dimension
                if len(vec_flat) != expected_dim:
                    print(f"Warning: Vector {i} has dimension {len(vec_flat)}, expected {expected_dim}")
                    # Pad or truncate to expected dimension
                    if len(vec_flat) < expected_dim:
                        vec_flat = np.pad(vec_flat, (0, expected_dim - len(vec_flat)), mode='constant')
                    else:
                        vec_flat = vec_flat[:expected_dim]
                
                vectors.append(vec_flat)
                
            embeddings = np.array(vectors) #为下一轮迭代做准备
            print(f"Final embeddings shape: {embeddings.shape}")
            all_nodes.append(nodes) 
            save_entities=copy.deepcopy(all_nodes)
            for layer in save_entities:
                if type(layer) != list :
                    if "vector" in layer.keys():
                        del layer["vector"]
                    continue
                for item in layer:
                    if "vector" in item.keys():
                        del item["vector"]
                    if len(layer)==1:
                        item['parent']='root'
            # check_test(all_entities)
            write_jsonl_force(save_entities, f"{WORKING_DIR}/all_entities.json")
            # check_test(all_nodes)            
            # stop if the number of deduplicated cluster is too small
            # if len(embeddings) <= 2:
            #     logging.info(f"[Stop Clustering at Layer{layer} with entity num {len(embeddings)}]")
            #     break
        if len(all_nodes[-1])!=1:
            temp_node={}
            cluster_nodes=all_nodes[-1]
            cluster_intern_relation=get_direct_relations(cluster_nodes,cluster_nodes,generate_relations)#默认为顶层，从下层找关系就是在generate_relations中
            describe=_pack_single_community_describe(cluster_nodes,cluster_intern_relation)
            hint_prompt=community_report_prompt.format(input_text=describe)
            # response = use_llm_func(hint_prompt,**llm_extra_kwargs)
            response = use_llm_func(hint_prompt)
            data = convert_response_to_json(response, hint_prompt)
            
            # Validate required fields and provide fallbacks for final layer
            if not data or not isinstance(data, dict):
                logger.warning(f"Invalid or empty response data for final layer. Using fallback.")
                data = {}
            
            # Ensure required fields exist with fallbacks
            entity_name = data.get('entity_name') or f"Root_Community_Layer_{layer}"
            entity_description = data.get('entity_description') or f"Root aggregate entity representing {len(cluster_nodes)} related entities"
            
            # Validate entity_name is not empty or just whitespace
            if not entity_name.strip():
                entity_name = f"Root_Community_Layer_{layer}"
                
            # Validate entity_description is not empty or just whitespace
            if not entity_description.strip():
                entity_description = f"Root aggregate entity representing {len(cluster_nodes)} related entities"
            
            data['level']=layer
            data['children']=[i['entity_name'] for i in cluster_nodes]
            data['source_id']= "|".join(set([i['source_id'] for i in cluster_nodes]))
            data['entity_name'] = entity_name
            data['entity_description'] = entity_description
            community_report[entity_name]=data
            
            temp_node['entity_name']=entity_name
            temp_node['description']=entity_description
            temp_node['source_id']="|".join(set(data['source_id'].split("|")))
            temp_node['entity_type']='community'
            temp_node['degree']=1
            temp_node['parent']='root'
            for i in cluster_nodes:
                i['parent']=entity_name
            
            all_nodes.append(temp_node)
        save_entities=copy.deepcopy(all_nodes)
        for layer in save_entities:
            if type(layer) != list :
                if "vector" in layer.keys():
                    del layer["vector"]
                continue
            for item in layer:
                if "vector" in item.keys():
                    del item["vector"]
                if len(layer)==1:
                    item['parent']='root'
        # check_test(all_entities)
        write_jsonl_force(save_entities, f"{WORKING_DIR}/all_entities.json")
        return all_nodes,generate_relations,community_report