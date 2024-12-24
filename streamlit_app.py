import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# BERTSearchEngine remains the same
class BERTSearchEngine:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.document_embeddings = {}

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
        return embedding[0]

    def index_documents(self, documents):
        for doc_id, doc_text in documents.items():
            self.document_embeddings[doc_id] = self.get_embedding(doc_text)

    def search(self, query, top_k=3):
        query_embedding = self.get_embedding(query)
        scores = {}
        for doc_id, doc_embedding in self.document_embeddings.items():
            similarity = 1 - cosine(query_embedding, doc_embedding)
            scores[doc_id] = similarity
        ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return ranked_results

class BERTVisualizer:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def get_attention_maps(self, text):
        inputs = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True, output_hidden_states=True)
        return {
            'tokens': self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]),
            'attention': outputs.attentions,
            'hidden_states': outputs.hidden_states
        }

    def visualize_attention(self, text, layer=0, head=0):
        attention_data = self.get_attention_maps(text)
        layer_attention = attention_data['attention'][layer][0][head].numpy()
        tokens = attention_data['tokens']
        
        fig = px.imshow(layer_attention,
                       labels=dict(x="Token", y="Token", color="Attention Weight"),
                       x=tokens,
                       y=tokens,
                       title=f'Attention Pattern - Layer {layer}, Head {head}')
        
        fig.update_xaxes(tickangle=45)
        fig.update_layout(height=600, width=800)
        return fig

    def visualize_multiple_heads(self, text):
        attention_data = self.get_attention_maps(text)
        tokens = attention_data['tokens']
        num_heads = attention_data['attention'][0][0].shape[0]  # Usually 12 for BERT base
        
        # Create subplots grid
        fig = go.Figure()
        
        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(num_heads)))
        
        for head_idx in range(num_heads):
            attention_weights = attention_data['attention'][0][0][head_idx].numpy()
            
            # Create heatmap for each head
            heatmap = go.Heatmap(
                z=attention_weights,
                x=tokens,
                y=tokens,
                colorscale='Viridis',
                showscale=True if head_idx == num_heads-1 else False,
                visible=False,
                name=f'Head {head_idx}'
            )
            fig.add_trace(heatmap)
        
        # Make first head visible by default
        fig.data[0].visible = True
        
        # Create buttons for head selection
        buttons = []
        for head_idx in range(num_heads):
            visibility = [False] * num_heads
            visibility[head_idx] = True
            buttons.append(
                dict(
                    method='restyle',
                    args=[{'visible': visibility}],
                    label=f'Head {head_idx}'
                )
            )
        
        fig.update_layout(
            updatemenus=[{
                'buttons': buttons,
                'direction': 'down',
                'showactive': True,
                'x': 0.1,
                'y': 1.15,
                'xanchor': 'left',
                'yanchor': 'top'
            }],
            title='Multiple Attention Heads Analysis<br>(Use dropdown to switch between heads)',
            height=700,
            width=800
        )
        
        fig.update_xaxes(tickangle=45)
        return fig

    def analyze_embedding_evolution(self, text):
        attention_data = self.get_attention_maps(text)
        hidden_states = attention_data['hidden_states']
        
        similarities = []
        for i in range(1, len(hidden_states)):
            prev_layer = hidden_states[i-1][0].mean(dim=0)
            curr_layer = hidden_states[i][0].mean(dim=0)
            similarity = torch.nn.functional.cosine_similarity(prev_layer.unsqueeze(0),
                                                            curr_layer.unsqueeze(0)).item()
            similarities.append(similarity)
        
        fig = px.line(x=list(range(1, len(similarities) + 1)), 
                     y=similarities,
                     markers=True,
                     labels={'x': 'Layer Transition', 'y': 'Cosine Similarity'},
                     title='Evolution of Embeddings Across Layers')
        
        fig.update_layout(height=500, width=800)
        return fig

    def analyze_head_importance(self, text):
        attention_data = self.get_attention_maps(text)
        attention_weights = attention_data['attention']
        
        head_importance = np.zeros((len(attention_weights), attention_weights[0].size(2)))
        for layer_idx, layer_attention in enumerate(attention_weights):
            for head_idx in range(layer_attention.size(2)):
                head_importance[layer_idx, head_idx] = layer_attention[0, head_idx].mean().item()
        
        fig = px.imshow(head_importance,
                       labels=dict(x="Head", y="Layer", color="Importance"),
                       title='Attention Head Importance Analysis')
        
        # Add numeric labels
        annotations = []
        for i in range(head_importance.shape[0]):
            for j in range(head_importance.shape[1]):
                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=f'{head_importance[i, j]:.2f}',
                        showarrow=False,
                        font=dict(size=8)
                    )
                )
        
        fig.update_layout(annotations=annotations, height=600, width=800)
        return fig

def attention_analysis_page():
    st.title("Attention Pattern Analysis")
    
    st.write("""
    This page provides comprehensive visualization of BERT's attention patterns,
    embedding evolution, and head importance analysis.
    """)
    
    sample_text = st.text_area("Enter text to analyze:", 
                              "The bank approved my loan application yesterday.")
    
    if sample_text:
        tab1, tab2, tab3, tab4 = st.tabs([
            "Single Head Analysis",
            "Multi-Head View",
            "Embedding Evolution",
            "Head Importance"
        ])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                layer = st.slider("Select Layer", 0, 11, 0)
            with col2:
                head = st.slider("Select Head", 0, 11, 0)
            fig = visualizer.visualize_attention(sample_text, layer, head)
            st.plotly_chart(fig)
        
        with tab2:
            st.subheader("Multi-Head Attention Visualization")
            fig = visualizer.visualize_multiple_heads(sample_text)
            st.plotly_chart(fig)
        
        with tab3:
            st.subheader("Embedding Evolution Analysis")
            fig = visualizer.analyze_embedding_evolution(sample_text)
            st.plotly_chart(fig)
            st.write("""
            This plot shows how token representations change across layers.
            Higher similarity indicates more stable representations,
            while lower similarity suggests significant transformation of the embeddings.
            """)
        
        with tab4:
            st.subheader("Head Importance Analysis")
            fig = visualizer.analyze_head_importance(sample_text)
            st.plotly_chart(fig)
            st.write("""
            This heatmap shows the average attention weight for each head in each layer.
            Darker colors indicate heads that contribute more strongly to the final representation.
            """)

# Initialize components and navigation
search_engine = BERTSearchEngine()
visualizer = BERTVisualizer()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", 
                       ["Attention Analysis"])

attention_analysis_page()