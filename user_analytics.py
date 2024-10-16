import streamlit as st
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
df = pd.read_excel("Hannie_Emotion_mapped.xlsx")
def basics():


    st.subheader("Active Energy Burned")
    fig1 = px.line(df, x="Timestamp", y="ActiveEnergyBurned")
    st.plotly_chart(fig1, use_container_width=True)


    st.subheader("Heart Rate")
    fig2 = px.line(df, x="Timestamp", y="HeartRate")
    st.plotly_chart(fig2, use_container_width=True)



    st.subheader("Physical Effort")
    fig3 = px.line(df, x="Timestamp", y="PhysicalEffort")
    st.plotly_chart(fig3, use_container_width=True)



    st.subheader("Basal Energy Burned") 
    fig4 = px.histogram(df, x= "BasalEnergyBurned", nbins = 10)
    fig4.update_layout(
    yaxis=dict(
        title="Total Energy Burnt",  # Y-axis title
        
        showgrid=True,         # Show gridlines
        zeroline=False,        # Hide the zero line
    ))
    st.plotly_chart(fig4, use_container_width=True)



def audio():
    threshold = 75
    colors = ['red' if val > threshold else 'blue' for val in df['EnvironmentalAudioExposure']]

    st.subheader("Environmental Audio Exposure Over Time")

    fig5 = go.Figure()
    fig5.add_trace(go.Bar(
        x=df['Timestamp'],
        y=df['EnvironmentalAudioExposure'],
        marker_color=colors,  # Set the colors based on the threshold
    ))

    # Update the layout
    fig5.update_layout(
       
        xaxis_title="Timestamp",
        yaxis_title="Environmental Audio Exposure (dB)",
        yaxis=dict(range=[0, 200]),
        plot_bgcolor='rgba(0,0,0,0)'
    )

    # Show the plot
    st.plotly_chart(fig5, use_container_width=True)


def walking():
    st.subheader("Distance Walking and Running")

    fig6 = px.line(df,x = 'Timestamp' , y= "DistanceWalkingRunning")
    st.plotly_chart(fig6, use_container_width=True)

    heart_grouped= df.groupby(['date'])['WalkingHeartRateAverage'].agg('mean')
    heart_grouped = pd.DataFrame(heart_grouped)
    heart_grouped.reset_index(inplace = True)
    heart_grouped['date'] = pd.to_datetime(heart_grouped['date'])


    st.subheader("Check your heart rate when you walk!")



    scatterfig = px.scatter(heart_grouped , x = heart_grouped['date'], y = 'WalkingHeartRateAverage')
    scatterfig.update_traces(marker_size=15)
    scatterfig.update_layout(xaxis=dict(type='category'))
    st.plotly_chart(scatterfig, use_container_width=True)



def go_to_page_user_analytics():


    st.write("Let us see your vitals")
    basics()
    audio()
    walking()