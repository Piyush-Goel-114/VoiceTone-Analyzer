import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import math

import dspy
from typing import Dict, List, Any

class RecommendationGenerator(dspy.Signature):
    """Generate actionable customer service recommendations based on conversation analysis"""
    
    overall_score = dspy.InputField(desc="Overall conversation satisfaction score (0-100)")
    risk_level = dspy.InputField(desc="Risk level classification (LOW, MEDIUM, HIGH, CRITICAL)")
    speaker_emotions = dspy.InputField(desc="Dictionary of speaker emotions and percentages")
    escalation_points = dspy.InputField(desc="Number and details of conversation escalation points")
    conversation_context = dspy.InputField(desc="Additional context about the conversation")
    
    recommendations = dspy.OutputField(desc="List of 3-5 actionable recommendations with emoji prefixes")

class ConversationSummarizer(dspy.Signature):
    """Generate comprehensive conversation summary with key metrics and insights"""
    
    analysis_data = dspy.InputField(desc="Complete analysis data including scores, emotions, and speaker info")
    conversation_flow = dspy.InputField(desc="Timeline and flow of the conversation")
    
    summary_json = dspy.OutputField(desc="JSON with total_speakers, primary_risk, customer_emotion, agent_performance, escalation_count, key_insights")

recommendation_generator = dspy.ChainOfThought(RecommendationGenerator)
conversation_summarizer = dspy.ChainOfThought(ConversationSummarizer)

st.set_page_config(
    page_title="Contact Scoring System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        border: 2px solid #495057;
    }
    
    .risk-critical {
        background: linear-gradient(135deg, #8b1538, #dc3545) !important;
        border-color: #721c24 !important;
        animation: pulse 2s infinite;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #d63384, #fd7e14) !important;
        border-color: #b02a5b !important;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #fd7e14, #ffc107) !important;
        border-color: #fd7e14 !important;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #198754, #20c997) !important;
        border-color: #146c43 !important;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Override Streamlit default alert colors */
    .stAlert > div {
        background-color: #343a40 !important;
        border: 2px solid #495057 !important;
        border-radius: 10px !important;
        color: white !important;
    }
    
    .stAlert > div > div {
        color: white !important;
    }
    
    /* Info alerts */
    .stAlert[data-baseweb="notification"] [data-testid="stMarkdownContainer"] {
        color: #ffffff !important;
    }
    
    /* Error alerts */
    .stAlert[kind="error"] > div {
        background-color: #721c24 !important;
        border-color: #8b1538 !important;
    }
    
    /* Warning alerts */
    .stAlert[kind="warning"] > div {
        background-color: #b02a37 !important;
        border-color: #d63384 !important;
    }
    
    /* Success alerts */
    .stAlert[kind="success"] > div {
        background-color: #146c43 !important;
        border-color: #198754 !important;
    }
    
    /* Info alerts */
    .stAlert[kind="info"] > div {
        background-color: #0f5132 !important;
        border-color: #198754 !important;
    }
    
    .escalation-point {
        background-color: #495057 !important;
        color: white !important;
        padding: 1rem;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
        border-radius: 8px;
        border: 1px solid #6c757d;
    }
    
    .timeline-event {
        background-color: #343a40 !important;
        color: white !important;
        padding: 1rem;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
        border-radius: 8px;
        border: 1px solid #495057;
    }
    
    .timeline-critical {
        border-left-color: #dc3545 !important;
        background-color: #721c24 !important;
        border-color: #8b1538 !important;
    }
    
    /* Dark theme for expanders */
    .streamlit-expanderHeader {
        background-color: #343a40 !important;
        border: 1px solid #495057 !important;
    }
    
    .streamlit-expanderContent {
        background-color: #2c3e50 !important;
        border: 1px solid #495057 !important;
    }
    
    /* Dark theme for metrics */
    [data-testid="metric-container"] {
        background-color: #343a40 !important;
        border: 1px solid #495057 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    [data-testid="metric-container"] > div {
        color: white !important;
    }
    
    /* Dark theme for file uploader */
    .stFileUploader > div {
        background-color: #343a40 !important;
        border: 2px dashed #6c757d !important;
        border-radius: 10px !important;
    }
    
    .stFileUploader label {
        color: white !important;
    }
    
    /* Dark theme for code blocks */
    .stCodeBlock {
        background-color: #2c3e50 !important;
        border: 1px solid #495057 !important;
    }
    
    /* Dark theme for download buttons */
    .stDownloadButton > button {
        background-color: #495057 !important;
        color: white !important;
        border: 1px solid #6c757d !important;
    }
    
    .stDownloadButton > button:hover {
        background-color: #6c757d !important;
        border-color: #adb5bd !important;
    }
</style>
""", unsafe_allow_html=True)

class ContactScorer:
    def __init__(self):
        self.emotion_weights = {
            'hap': 100,
            'neu': 75,
            'sad': 40,
            'ang': 10
        }
        
        self.emotion_colors = {
            'ang': '#dc3545',
            'hap': '#28a745',
            'sad': '#6f42c1',
            'neu': '#6c757d'
        }
    
    def calculate_segment_score(self, segment):
        """Calculate score for individual segment"""
        emotion = segment.get('emotion', 'neu')
        confidence = segment.get('emotion_confidence', 0)
        
        base_score = self.emotion_weights.get(emotion, 50)
        
        if emotion in ['ang', 'sad']:
            return base_score * (2 - confidence)
        
        return base_score * confidence
    
    def analyze_speaker(self, segments):
        """Analyze individual speaker data"""
        if not segments:
            return {}
            
        stats = {
            'total_segments': len(segments),
            'emotion_counts': {'ang': 0, 'hap': 0, 'sad': 0, 'neu': 0},
            'emotion_percentages': {},
            'average_confidence': 0,
            'weighted_score': 0,
            'dominant_emotion': '',
            'escalation_points': [],
            'average_duration': 0,
            'total_duration': 0
        }
        
        total_confidence = 0
        total_duration = 0
        score_sum = 0
        
        for segment in segments:
            emotion = segment.get('emotion', 'neu')
            confidence = segment.get('emotion_confidence', 0)
            duration = segment.get('duration', 0)
            
            stats['emotion_counts'][emotion] += 1
            total_confidence += confidence
            total_duration += duration
            
            segment_score = self.calculate_segment_score(segment)
            score_sum += segment_score
            
            # Track escalation points
            if emotion == 'ang' and confidence > 0.7:
                stats['escalation_points'].append({
                    'time': segment.get('start_time', 0),
                    'intensity': confidence,
                    'transcript': segment.get('transcript', ''),
                    'duration': duration,
                    'emotion': emotion
                })
        
        stats['average_confidence'] = total_confidence / len(segments)
        stats['average_duration'] = total_duration / len(segments)
        stats['total_duration'] = total_duration
        stats['weighted_score'] = score_sum / len(segments)
        
        # Calculate percentages
        for emotion in stats['emotion_counts']:
            stats['emotion_percentages'][emotion] = (stats['emotion_counts'][emotion] / len(segments)) * 100
        
        # Find dominant emotion
        stats['dominant_emotion'] = max(stats['emotion_counts'], key=stats['emotion_counts'].get)
        
        return stats
    
    def analyze_conversation(self, data):
        """Main analysis function"""
        if not data:
            return {}
            
        speakers = list(data.keys())
        analysis = {
            'overall_score': 0,
            'risk_level': 'Low',
            'speaker_analysis': {},
            'recommendations': [],
            'timeline': [],
            'summary': {}
        }
        
        total_segments = 0
        total_score = 0

        new_data = {"SPEAKER": []}

        for speaker in speakers:
            new_data["SPEAKER"].extend(data[speaker])
        new_data["SPEAKER"].sort(key=lambda x: x.get('start_time', 0))

        for speaker in ["SPEAKER"]:
            segments = new_data[speaker]
            speaker_stats = self.analyze_speaker(segments)
            analysis['speaker_analysis'][speaker] = speaker_stats
            
            total_segments += len(segments)
            total_score += speaker_stats['weighted_score'] * len(segments)
            
            # Add timeline events
            for segment in segments:
                if segment.get('emotion') == 'ang' and segment.get('emotion_confidence', 0) > 0.8:
                    start_time = segment.get('start_time', 0)
                    minutes = int(start_time // 60)
                    seconds = int(start_time % 60)
                    
                    analysis['timeline'].append({
                        'time': f"{minutes}:{seconds:02d}",
                        'speaker': speaker,
                        'event': 'High Anger Detected',
                        'transcript': segment.get('transcript', ''),
                        'severity': 'high',
                        'confidence': segment.get('emotion_confidence', 0)
                    })
        
        if total_segments > 0:
            analysis['overall_score'] = round(total_score / total_segments)
        
        analysis['risk_level'] = self.get_risk_level(analysis['overall_score'])
        analysis['recommendations'] = self.generate_recommendations(analysis)
        analysis['summary'] = self.generate_summary(analysis)
        
        return analysis
    
    def get_risk_level(self, score):
        """Determine risk level based on score"""
        if score >= 80:
            return 'Low'
        elif score >= 60:
            return 'Medium'
        elif score >= 40:
            return 'High'
        else:
            return 'Critical'
    
    def generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate ML-powered actionable recommendations"""
        
        # Prepare structured input for the ML model
        speaker_emotions = {}
        escalation_details = []

        lm = dspy.LM('groq/gemma2-9b-it', temperature=0.3, api_key="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")
        dspy.settings.configure(lm=lm)
        
        for speaker, stats in analysis.get('speaker_analysis', {}).items():
            speaker_emotions[speaker] = {
                'emotions': stats.get('emotion_percentages', {}),
                'dominant': stats.get('dominant_emotion', 'neutral'),
                'weighted_score': stats.get('weighted_score', 50)
            }
            escalation_details.extend(stats.get('escalation_points', []))
        
        conversation_context = f"""
        Overall Score: {analysis.get('overall_score', 50)}
        Risk Level: {analysis.get('risk_level', 'MEDIUM')}
        Total Escalations: {len(escalation_details)}
        Speaker Count: {len(analysis.get('speaker_analysis', {}))}
        """
        
        try:
            result = recommendation_generator(
                overall_score=str(analysis.get('overall_score', 50)),
                risk_level=analysis.get('risk_level', 'MEDIUM'),
                speaker_emotions=str(speaker_emotions),
                escalation_points=str(escalation_details),
                conversation_context=conversation_context
            )
            
            recommendations = []
            lines = result.recommendations.split('\n')
            
            for line in lines:
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line.startswith('*') or '📈' in line or '🚨' in line or '✅' in line):
                    clean_line = line.lstrip('-•* ').strip()
                    if clean_line:
                        if not any(emoji in clean_line for emoji in ['🚨', '📈', '😤', '⚡', '😐', '✅', '💡']):
                            clean_line = f"💡 {clean_line}"
                        recommendations.append(clean_line)
            
            if recommendations:
                return recommendations[:20]
            else:
                return self._fallback_recommendations(analysis)
                
        except Exception as e:
            print(f"ML recommendation generation failed: {e}")
            return self._fallback_recommendations(analysis)
        
    def _fallback_recommendations(self, analysis):
        """Generate actionable recommendations"""
        recommendations = []
        
        if analysis['overall_score'] < 50:
            recommendations.append('🚨 Immediate follow-up required - Customer shows high dissatisfaction')
            recommendations.append('📈 Consider escalating to senior customer service representative')
        
        for speaker, stats in analysis['speaker_analysis'].items():
            if stats['emotion_percentages'].get('ang', 0) > 40:
                recommendations.append(f'😤 {speaker} shows significant anger - Implement de-escalation strategies')
            
            if len(stats['escalation_points']) > 3:
                recommendations.append(f'⚡ {speaker} has multiple escalation points - Review conversation for service improvements')
            
            if stats['emotion_percentages'].get('hap', 0) < 10:
                recommendations.append(f'😐 {speaker} shows minimal positive sentiment - Focus on relationship building')
        
        if not recommendations:
            recommendations.append('✅ Conversation shows generally positive sentiment - Standard follow-up appropriate')
        
        return recommendations
    
    def generate_summary(self, analysis):
        """Generate conversation summary"""
        speakers = list(analysis['speaker_analysis'].keys())
        customer_speaker = next((s for s in speakers if s != 'SPEAKER_00'), speakers[0] if speakers else None)
        agent_speaker = next((s for s in speakers if s == 'SPEAKER_00'), speakers[1] if len(speakers) > 1 else None)
        
        escalation_count = sum(len(stats['escalation_points']) for stats in analysis['speaker_analysis'].values())
        
        return {
            'total_speakers': len(speakers),
            'primary_risk': analysis['risk_level'],
            'customer_emotion': analysis['speaker_analysis'].get(customer_speaker, {}).get('dominant_emotion', 'Unknown') if customer_speaker else 'Unknown',
            'agent_performance': analysis['speaker_analysis'].get(agent_speaker, {}).get('weighted_score', 0) if agent_speaker else 0,
            'escalation_count': escalation_count,
            'customer_speaker': customer_speaker,
            'agent_speaker': agent_speaker
        }

def main():
    st.title("🎯 Contact Scoring System")
    st.markdown("### AI-Powered Emotion Analysis & Customer Interaction Scoring")
    
    scorer = ContactScorer()
    
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. Upload your emotion analysis JSON file
        2. View comprehensive scoring analysis
        3. Review recommendations and timeline
        4. Download detailed report
        """)
        
        st.header("📊 Scoring Criteria")
        st.markdown("""
        **Emotion Weights:**
        - 😊 Happy: 100 points
        - 😐 Neutral: 75 points  
        - 😢 Sad: 40 points
        - 😡 Anger: 10 points
        
        **Risk Levels:**
        - 80-100: Low Risk
        - 60-79: Medium Risk
        - 40-59: High Risk
        - 0-39: Critical Risk
        """)
    
    uploaded_file = st.file_uploader(
        "Choose your emotion analysis JSON file",
        type=['json'],
        help="Upload the hf_emotion_enhanced_results.json file"
    )
    
    if uploaded_file is not None:
        try:
            emotion_data = json.loads(uploaded_file.getvalue().decode('utf-8'))
            with st.spinner('Processing emotion data and calculating scores...'):
                analysis = scorer.analyze_conversation(emotion_data)
            if not analysis:
                st.error("Unable to process the uploaded file. Please check the format.")
                return
            display_results(analysis, scorer)
            
        except json.JSONDecodeError:
            st.error("Error parsing JSON file. Please ensure it's a valid JSON file.")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        st.markdown("""
        <div style="background-color: #0f5132; color: white; padding: 1rem; margin: 0.5rem 0; border-radius: 8px; border-left: 4px solid #198754; border: 1px solid #198754;">
            <strong>👆 Upload your emotion analysis JSON file to get started</strong>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("📋 Expected JSON Format"):
            st.markdown("""
            <div style="background-color: #343a40; padding: 1rem; border-radius: 8px; border: 1px solid #495057;">
            """, unsafe_allow_html=True)
            st.code('''
{
    "SPEAKER_01": [
        {
            "start_time": 1.28,
            "end_time": 2.51,
            "transcript": "who gave me a user voucher.",
            "duration": 1.23,
            "emotion": "neu",
            "emotion_confidence": 0.7976725101470947,
            "emotion_scores": {
                "neu": 0.7976725101470947,
                "ang": 0.17265382409095764,
                "sad": 0.017911504954099655,
                "hap": 0.0117621636018157
            }
        }
    ]
}
            ''', language='json')
            st.markdown("</div>", unsafe_allow_html=True)

def display_results(analysis, scorer):
    """Display comprehensive analysis results"""
    
    # Summary metrics
    st.header("📊 Contact Score Summary")
    
    cols = st.columns(4)
    
    risk_class = f"risk-{analysis['risk_level'].lower()}"
    
    with cols[0]:
        st.markdown(f"""
        <div class="metric-container {risk_class}">
            <h3>Overall Score</h3>
            <h1>{analysis['overall_score']}</h1>
            <p>Customer Satisfaction Index</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown(f"""
        <div class="metric-container {risk_class}">
            <h3>Risk Level</h3>
            <h1>{analysis['risk_level']}</h1>
            <p>Escalation Probability</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown(f"""
        <div class="metric-container">
            <h3>Escalations</h3>
            <h1>{analysis['summary']['escalation_count']}</h1>
            <p>Critical Moments</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[3]:
        st.markdown(f"""
        <div class="metric-container">
            <h3>Speakers</h3>
            <h1>{analysis['summary']['total_speakers']}</h1>
            <p>Conversation Participants</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed speaker analysis
    st.header("🔍 Detailed Speaker Analysis")
    
    for speaker, stats in analysis['speaker_analysis'].items():
        speaker_type = ""
        
        with st.expander(f"👤 {speaker} {speaker_type} - Score: {round(stats['weighted_score'])}/100", expanded=True):
            
            # Speaker metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Segments", stats['total_segments'])
            with col2:
                st.metric("Avg Confidence", f"{round(stats['average_confidence'] * 100)}%")
            with col3:
                st.metric("Dominant Emotion", stats['dominant_emotion'].upper())
            with col4:
                st.metric("Avg Duration", f"{round(stats['average_duration'], 1)}s")
            
            # Emotion breakdown chart
            emotion_df = pd.DataFrame([
                {'Emotion': emotion.upper(), 'Percentage': percentage, 'Count': stats['emotion_counts'][emotion]}
                for emotion, percentage in stats['emotion_percentages'].items()
                if percentage > 0
            ])
            
            if not emotion_df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_pie = px.pie(
                        emotion_df, 
                        values='Percentage', 
                        names='Emotion',
                        title='Emotion Distribution',
                        color='Emotion',
                        color_discrete_map={
                            'ANG': '#dc3545',
                            'HAP': '#28a745', 
                            'SAD': '#6f42c1',
                            'NEU': '#6c757d'
                        }
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    fig_bar = px.bar(
                        emotion_df,
                        x='Emotion',
                        y='Percentage',
                        title='Emotion Percentages',
                        color='Emotion',
                        color_discrete_map={
                            'ANG': '#dc3545',
                            'HAP': '#28a745',
                            'SAD': '#6f42c1', 
                            'NEU': '#6c757d'
                        }
                    )
                    fig_bar.update_layout(showlegend=False)
                    st.plotly_chart(fig_bar, use_container_width=True)
            
            # Escalation points
            if stats['escalation_points']:
                st.subheader(f"🚨 Escalation Points ({len(stats['escalation_points'])})")
                
                for point in stats['escalation_points']:
                    minutes = int(point['time'] // 60)
                    seconds = int(point['time'] % 60)
                    emotion_df = point.get('emotion', 'Unknown')
                    
                    st.markdown(f"""
                    <div class="escalation-point">
                        <strong>{minutes}:{seconds:02d}</strong> 
                        <strong>{emotion_df}</strong>
                        (Intensity: {round(point['intensity'] * 100)}%) - "{point['transcript']}"
                    </div>
                    """, unsafe_allow_html=True)
    
    # Timeline analysis
    if analysis['timeline']:
        st.header("⏰ Critical Timeline Events")
        
        for event in analysis['timeline']:
            severity_class = "timeline-critical" if event['severity'] == 'high' else ""
            
            st.markdown(f"""
            <div class="timeline-event {severity_class}">
                <strong>{event['time']} - {event['speaker']}</strong>: {event['event']}
                <br>
                <em>Confidence: {round(event['confidence'] * 100)}%</em>
                <br>
                <em>"{event['transcript']}"</em>
            </div>
            """, unsafe_allow_html=True)
    
    # Recommendations
    st.header("🎯 Action Recommendations")
    
    for i, recommendation in enumerate(analysis['recommendations'], 1):
        if '🚨' in recommendation or 'Critical' in recommendation:
            st.markdown(f"""
            <div style="background-color: #721c24; color: white; padding: 1rem; margin: 0.5rem 0; border-radius: 8px; border-left: 4px solid #dc3545; border: 1px solid #8b1538;">
                <strong>{i}. {recommendation}</strong>
            </div>
            """, unsafe_allow_html=True)
        elif '⚡' in recommendation or '😤' in recommendation:
            st.markdown(f"""
            <div style="background-color: #b02a37; color: white; padding: 1rem; margin: 0.5rem 0; border-radius: 8px; border-left: 4px solid #fd7e14; border: 1px solid #d63384;">
                <strong>{i}. {recommendation}</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #0f5132; color: white; padding: 1rem; margin: 0.5rem 0; border-radius: 8px; border-left: 4px solid #198754; border: 1px solid #198754;">
                <strong>{i}. {recommendation}</strong>
            </div>
            """, unsafe_allow_html=True)
    
    # Download section
    st.header("📥 Export Results")
    
    # Create detailed report
    report_data = {
        'analysis_timestamp': datetime.now().isoformat(),
        'overall_metrics': {
            'score': analysis['overall_score'],
            'risk_level': analysis['risk_level'],
            'total_speakers': analysis['summary']['total_speakers'],
            'escalation_count': analysis['summary']['escalation_count']
        },
        'speaker_analysis': analysis['speaker_analysis'],
        'recommendations': analysis['recommendations'],
        'timeline_events': analysis['timeline']
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📊 Download Detailed JSON Report",
            data=json.dumps(report_data, indent=2),
            file_name=f"contact_score_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col2:
        # Create CSV summary
        csv_data = []
        for speaker, stats in analysis['speaker_analysis'].items():
            csv_data.append({
                'Speaker': speaker,
                'Score': round(stats['weighted_score']),
                'Total_Segments': stats['total_segments'],
                'Dominant_Emotion': stats['dominant_emotion'],
                'Anger_Percentage': round(stats['emotion_percentages'].get('ang', 0), 1),
                'Happy_Percentage': round(stats['emotion_percentages'].get('hap', 0), 1),
                'Escalation_Points': len(stats['escalation_points']),
                'Average_Confidence': round(stats['average_confidence'] * 100, 1)
            })
        
        csv_df = pd.DataFrame(csv_data)
        
        st.download_button(
            label="📈 Download CSV Summary",
            data=csv_df.to_csv(index=False),
            file_name=f"contact_score_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()