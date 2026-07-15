# src/visuals.py
#
# Note: charts are built with plotly.graph_objects (not plotly.express) and
# every plot_* function is cached with @st.cache_data. Repeated Plotly figure
# creation across many Streamlit reruns segfaults the Python process in this
# environment (a native crash, not a catchable exception) — caching means a
# rerun that doesn't actually change a chart's inputs reuses the existing
# figure instead of building a new one, which is what keeps a real session
# usable. See app.py's section-navigation comment for the full explanation.
import streamlit as st
import plotly.graph_objects as go

# Categorical palette (fixed order — never cycled/reassigned). Teal/magenta/
# violet/amber to match the dashboard's dark, glowing hero banner.
CATEGORICAL = ['#0d9488', '#db2777', '#7c3aed', '#d97706', '#0891b2', '#e11d48', '#65a30d', '#c026d3']
CLASS_COLORS = {'Concordant': '#0d9488', 'Conflicting': '#db2777'}
SEQUENTIAL = ['#ccfbf1', '#5eead4', '#2dd4bf', '#0d9488', '#0f766e', '#134e4a']
DIVERGING = ['#134e4a', '#0d9488', '#334155', '#db2777', '#831843']
STATUS = {'good': '#0ca30c', 'warning': '#fab219', 'serious': '#ec835a', 'critical': '#d03b3b'}

TEMPLATE = 'plotly_dark'
FONT = dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=13, color='#dbe4f0')

# Dark surfaces the dashboard's dark theme renders charts on
PLOT_BG = '#0b1220'
GRID_COLOR = 'rgba(148,163,184,0.15)'
AXIS_COLOR = 'rgba(148,163,184,0.35)'


def _style(fig, title=None, height=380):
    fig.update_layout(
        template=TEMPLATE,
        font=FONT,
        title=dict(text=title, font=dict(size=16, color='#f4f7fb')) if title else None,
        height=height,
        margin=dict(l=40, r=30, t=50 if title else 20, b=40),
        plot_bgcolor=PLOT_BG,
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(color='#dbe4f0')),
        hoverlabel=dict(bgcolor='#101a2e', font_size=13, font_color='#f4f7fb'),
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=AXIS_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=AXIS_COLOR)
    return fig


@st.cache_data
def plot_histogram(df, col, title=None):
    fig = go.Figure(data=go.Histogram(
        x=df[col], nbinsx=40, marker=dict(color=CATEGORICAL[0], line_width=0), opacity=0.9
    ))
    fig.update_xaxes(title_text=col)
    fig.update_yaxes(title_text='count')
    return _style(fig, title or f'Distribution of {col}')


@st.cache_data
def plot_box(df, col, title=None):
    fig = go.Figure(data=go.Box(y=df[col], name=col, marker=dict(color=CATEGORICAL[0]), boxpoints='outliers'))
    fig.update_yaxes(title_text=col)
    return _style(fig, title or f'Outlier Check: {col}', height=380)


@st.cache_data
def plot_count(df, col, class_col='class', title=None):
    plot_df = df.copy()
    order = plot_df[col].value_counts().index[:12]
    plot_df = plot_df[plot_df[col].isin(order)]

    fig = go.Figure()
    if class_col in plot_df.columns:
        plot_df['Class'] = plot_df[class_col].map({0: 'Concordant', 1: 'Conflicting'})
        for class_name in ['Concordant', 'Conflicting']:
            counts = plot_df[plot_df['Class'] == class_name][col].value_counts().reindex(order, fill_value=0)
            fig.add_trace(go.Bar(x=list(order), y=counts.values, name=class_name, marker_color=CLASS_COLORS[class_name]))
        fig.update_layout(barmode='group')
    else:
        counts = plot_df[col].value_counts().reindex(order, fill_value=0)
        fig.add_trace(go.Bar(x=list(order), y=counts.values, marker_color=CATEGORICAL[0]))
    fig.update_xaxes(title_text=col)
    fig.update_yaxes(title_text='count')
    return _style(fig, title or f'Count of {col}')


@st.cache_data
def plot_correlation_heatmap(df, title='Correlation Heatmap'):
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr = numeric_df.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale=[[0, DIVERGING[0]], [0.25, DIVERGING[1]], [0.5, DIVERGING[2]], [0.75, DIVERGING[3]], [1, DIVERGING[4]]],
        zmid=0, zmin=-1, zmax=1, colorbar=dict(title='r'),
        hovertemplate='%{x} × %{y}<br>r = %{z:.2f}<extra></extra>'
    ))
    return _style(fig, title, height=520)


@st.cache_data
def plot_scatter(df, x, y, class_col='class', title=None):
    plot_df = df.copy()
    fig = go.Figure()
    if class_col in plot_df.columns:
        plot_df['Class'] = plot_df[class_col].map({0: 'Concordant', 1: 'Conflicting'})
        for class_name in ['Concordant', 'Conflicting']:
            subset = plot_df[plot_df['Class'] == class_name]
            fig.add_trace(go.Scatter(
                x=subset[x], y=subset[y], mode='markers', name=class_name,
                marker=dict(size=6, color=CLASS_COLORS[class_name], opacity=0.5)
            ))
    else:
        fig.add_trace(go.Scatter(
            x=plot_df[x], y=plot_df[y], mode='markers',
            marker=dict(size=6, color=CATEGORICAL[0], opacity=0.5)
        ))
    fig.update_xaxes(title_text=x)
    fig.update_yaxes(title_text=y)
    return _style(fig, title or f'{x} vs {y}')


@st.cache_data
def plot_bar_by_category(df, category_col, value_col, agg='mean', title=None):
    grouped = df.groupby(category_col)[value_col].agg(agg).reset_index()
    fig = go.Figure(data=go.Bar(
        x=grouped[category_col], y=grouped[value_col], marker=dict(color=CATEGORICAL[0], line_width=0)
    ))
    fig.update_xaxes(title_text=category_col)
    fig.update_yaxes(title_text=value_col)
    return _style(fig, title or f'{agg.title()} {value_col} by {category_col}')


@st.cache_data
def plot_class_distribution(df, class_col='class', title='Target Variable Distribution'):
    plot_df = df.copy()
    plot_df['Class'] = plot_df[class_col].map({0: 'Concordant', 1: 'Conflicting'})
    counts = plot_df['Class'].value_counts().reindex(['Concordant', 'Conflicting'])
    fig = go.Figure(data=[go.Pie(
        labels=counts.index, values=counts.values, hole=0.55,
        marker=dict(colors=[CLASS_COLORS['Concordant'], CLASS_COLORS['Conflicting']]),
        textinfo='label+percent', textfont=dict(size=14),
    )])
    return _style(fig, title, height=380)


def plot_confidence_gauge(probability, title='Conflict Probability'):
    if probability < 0.34:
        bar_color = STATUS['good']
    elif probability < 0.5:
        bar_color = STATUS['warning']
    elif probability < 0.66:
        bar_color = STATUS['serious']
    else:
        bar_color = STATUS['critical']

    fig = go.Figure(go.Indicator(
        mode='gauge+number+delta',
        value=probability * 100,
        number=dict(suffix='%', font=dict(size=40, color='#f4f7fb')),
        delta=dict(reference=25.2, increasing=dict(color=STATUS['critical']), decreasing=dict(color=STATUS['good'])),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor='#94a3b8', tickfont=dict(color='#94a3b8')),
            bar=dict(color=bar_color, thickness=0.3),
            bgcolor=PLOT_BG,
            bordercolor='rgba(148,163,184,0.25)',
            borderwidth=1,
            steps=[
                dict(range=[0, 34], color='rgba(12,163,12,0.18)'),
                dict(range=[34, 50], color='rgba(250,178,25,0.18)'),
                dict(range=[50, 66], color='rgba(236,131,90,0.18)'),
                dict(range=[66, 100], color='rgba(208,59,59,0.20)'),
            ],
            threshold=dict(line=dict(color='#f4f7fb', width=3), thickness=0.85, value=probability * 100)
        ),
        title=dict(text=title, font=dict(size=16, color='#f4f7fb'))
    ))
    fig.update_layout(
        template=TEMPLATE, font=FONT, height=320,
        margin=dict(l=30, r=30, t=60, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        transition=dict(duration=800, easing='cubic-in-out'),
    )
    return fig
