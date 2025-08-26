import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import pandas as pd 
import streamlit as st
import sys

pd.set_option('display.max_columns', 4)

def view_collections(dir):
    st.markdown("### DB Path: %s" % dir)

    try:
        client = chromadb.PersistentClient(path=dir)
        collections = client.list_collections()
        
        st.success(f"✅ Connected to ChromaDB! Found {len(collections)} collections.")
        
        if not collections:
            st.warning("No collections found in this ChromaDB.")
            return

        st.header("Collections")

        for collection in collections:
            data = collection.get()

            ids = data['ids']
            embeddings = data["embeddings"]
            metadata = data["metadatas"]
            documents = data["documents"]

            st.markdown("### Collection: **%s**" % collection.name)
            st.markdown(f"**Entities:** {len(ids) if ids else 0}")
            
            if ids and len(ids) > 0:
                # Create a more readable dataframe
                display_data = {
                    'ID': ids[:100],  # Show first 100 entities
                    'Entity Name': [meta.get('entity_name', 'Unknown') if meta else 'Unknown' for meta in (metadata[:100] if metadata else [])],
                    'Level': [meta.get('level', 'Unknown') if meta else 'Unknown' for meta in (metadata[:100] if metadata else [])],
                    'Parent': [meta.get('parent', 'Unknown') if meta else 'Unknown' for meta in (metadata[:100] if metadata else [])],
                    'Document': [doc[:100] + '...' if doc and len(doc) > 100 else doc for doc in (documents[:100] if documents else [])]
                }
                
                df = pd.DataFrame.from_dict(display_data)
                st.dataframe(df, use_container_width=True)
                
                if len(ids) > 100:
                    st.info(f"Showing first 100 entities out of {len(ids)} total entities.")
            else:
                st.warning("Collection is empty.")
                
    except Exception as e:
        st.error(f"❌ Error connecting to ChromaDB: {e}")

if __name__ == "__main__":
    st.title("ChromaDB Viewer")
    
    # Use Streamlit input instead of argparse
    default_path = "data/mix/chromadb"
    if len(sys.argv) > 1:
        default_path = sys.argv[1]
    
    db_path = st.text_input("ChromaDB Path:", value=default_path)
    
    if st.button("View Database") or db_path:
        if db_path:
            view_collections(db_path)
        else:
            st.warning("Please enter a ChromaDB path.")