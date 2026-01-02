#!/usr/bin/env python3
"""
Visualization Utilities
Generate interactive Plotly charts for malware analysis
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

def create_confusion_matrix(cm, labels=['Clean', 'Malware']):
    """Create interactive confusion matrix heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Reds',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        showscale=True
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        width=500,
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(20,20,30,0.5)',
        font=dict(color='#00ff00', size=12)
    )
    
    return fig

def create_roc_curve(fpr, tpr, auc_score):
    """Create ROC curve"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC (AUC = {auc_score:.3f})',
        line=dict(color='#ff0066', width=3)
    ))
    
    # Diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='#666', width=1, dash='dash')
    ))
    
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=550,
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(20,20,30,0.5)',
        font=dict(color='#00ff00', size=12),
        showlegend=True
    )
    
    return fig

def create_feature_importance(feature_names, importances, top_n=15):
    """Create feature importance bar chart"""
    # Sort and get top N features
    sorted_idx = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in sorted_idx]
    top_importances = importances[sorted_idx]
    
    fig = go.Figure(data=[
        go.Bar(
            x=top_importances,
            y=top_features,
            orientation='h',
            marker=dict(
                color=top_importances,
                colorscale='Viridis',
                showscale=True
            )
        )
    ])
    
    fig.update_layout(
        title=f'Top {top_n} Feature Importances',
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(20,20,30,0.5)',
        font=dict(color='#00ff00', size=12)
    )
    
    return fig

def create_model_comparison(accuracies):
    """Create model accuracy comparison chart"""
    models = list(accuracies.keys())
    scores = list(accuracies.values())
    
    colors = ['#00ff00', '#ff0066', '#00ccff', '#ffaa00']
    
    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=scores,
            text=[f'{s:.2f}%' for s in scores],
            textposition='auto',
            marker=dict(color=colors[:len(models)])
        )
    ])
    
    fig.update_layout(
        title='Model Accuracy Comparison',
        xaxis_title='Model',
        yaxis_title='Accuracy (%)',
        yaxis=dict(range=[90, 100]),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(20,20,30,0.5)',
        font=dict(color='#00ff00', size=12)
    )
    
    return fig

def create_threat_gauge(probability):
    """Create threat level gauge"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Malware Risk Score", 'font': {'size': 24, 'color': '#00ff00'}},
        number={'suffix': "%", 'font': {'size': 40, 'color': '#ff0066'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#00ff00"},
            'bar': {'color': "#ff0066"},
            'bgcolor': "rgba(20,20,30,0.5)",
            'borderwidth': 2,
            'bordercolor': "#00ff00",
            'steps': [
                {'range': [0, 25], 'color': 'rgba(0, 255, 0, 0.3)'},
                {'range': [25, 50], 'color': 'rgba(255, 255, 0, 0.3)'},
                {'range': [50, 75], 'color': 'rgba(255, 165, 0, 0.3)'},
                {'range': [75, 100], 'color': 'rgba(255, 0, 0, 0.3)'}],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "#00ff00", 'family': "Courier New"},
        height=350
    )
    
    return fig

def create_prediction_breakdown(individual_probs):
    """Create individual model predictions breakdown"""
    models = list(individual_probs.keys())
    probabilities = [p * 100 for p in individual_probs.values()]
    
    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=probabilities,
            text=[f'{p:.1f}%' for p in probabilities],
            textposition='auto',
            marker=dict(
                color=probabilities,
                colorscale='RdYlGn_r',
                showscale=False
            )
        )
    ])
    
    fig.update_layout(
        title='Individual Model Predictions',
        xaxis_title='Model',
        yaxis_title='Malware Probability (%)',
        yaxis=dict(range=[0, 100]),
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(20,20,30,0.5)',
        font=dict(color='#00ff00', size=12)
    )
    
    return fig
