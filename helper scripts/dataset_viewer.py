#!/usr/bin/env python
"""
Dataset Viewer

A web application for visualizing span-based voting datasets.
Displays both NER sequence labeling and span-based formats with entity highlighting.

Usage:
    python dataset_viewer.py
    python dataset_viewer.py --port 5000 --debug
"""

import os
import json
import argparse
from pathlib import Path
from flask import Flask, render_template_string, request
from collections import Counter

app = Flask(__name__)

def get_entity_color(label):
    """Get color for entity labels."""
    colors = {
        'SUBJECT': '#FFE4B5',
        'VOTING': '#98FB98',
        'VOTER-FAVOR': '#32CD32',
        'VOTER-AGAINST': '#DC143C',
        'VOTER-ABSTENTION': '#FFA500',
        'VOTER-ABSENT': '#FFD700',
        'COUNTING-UNANIMITY': '#87CEEB',
        'COUNTING-MAJORITY': '#B0C4DE'
    }
    return colors.get(label, '#F0F0F0')

def discover_datasets(base_dir):
    """Discover available datasets."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return {}

    datasets = {}

    # Discover NER datasets
    ner_dir = base_path / 'ner_datasets'
    if ner_dir.exists():
        for lang_dir in ner_dir.iterdir():
            if lang_dir.is_dir():
                language = lang_dir.name
                full_dataset = lang_dir / f'full_dataset_{language}_ner.jsonl'

                if full_dataset.exists():
                    with open(full_dataset, 'r', encoding='utf-8') as f:
                        count = sum(1 for line in f if line.strip())

                    datasets[f'NER ({language.upper()})'] = {
                        'type': 'ner',
                        'file_path': str(full_dataset),
                        'count': count,
                        'language': language
                    }

    # Discover span datasets
    span_dir = base_path / 'span_datasets'
    if span_dir.exists():
        for lang_dir in span_dir.iterdir():
            if lang_dir.is_dir():
                language = lang_dir.name
                full_dataset = lang_dir / f'full_dataset_{language}_spans.json'

                if full_dataset.exists():
                    with open(full_dataset, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        count = len(data)

                    datasets[f'Span ({language.upper()})'] = {
                        'type': 'span',
                        'file_path': str(full_dataset),
                        'count': count,
                        'language': language
                    }

    return datasets

def get_dataset_statistics(datasets):
    """Calculate dataset statistics."""
    stats = {
        'total_ner_examples': 0,
        'total_span_examples': 0,
        'languages': set(),
        'entity_distribution': Counter()
    }

    for dataset_info in datasets.values():
        if dataset_info['type'] == 'ner':
            stats['total_ner_examples'] += dataset_info['count']
            # Count entities in NER data
            try:
                with open(dataset_info['file_path'], 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            example = json.loads(line)
                            for tag in example.get('labels', []):
                                if tag.startswith('B-'):
                                    entity_type = tag[2:]
                                    stats['entity_distribution'][entity_type] += 1
            except:
                pass

        elif dataset_info['type'] == 'span':
            stats['total_span_examples'] += dataset_info['count']

        stats['languages'].add(dataset_info['language'])

    stats['languages'] = sorted(list(stats['languages']))
    return stats

def render_ner_tokens(tokens, labels):
    """Render NER tokens with BIO tags."""
    if len(tokens) != len(labels):
        return f"<span class='error'>Mismatch: {len(tokens)} tokens, {len(labels)} labels</span>"

    html_parts = []
    current_entity = None
    current_tokens = []

    for token, label in zip(tokens, labels):
        # Clean token to avoid display issues
        clean_token = token.strip()
        if not clean_token:
            continue

        if label == 'O':
            if current_entity:
                html_parts.append(render_entity_span(current_tokens, current_entity))
                current_entity = None
                current_tokens = []
            html_parts.append(f'<span class="token">{escape_html(clean_token)}</span> ')

        elif label.startswith('B-'):
            if current_entity:
                html_parts.append(render_entity_span(current_tokens, current_entity))
            current_entity = label[2:]
            current_tokens = [clean_token]

        elif label.startswith('I-'):
            entity_type = label[2:]
            if current_entity == entity_type:
                current_tokens.append(clean_token)
            else:
                if current_entity:
                    html_parts.append(render_entity_span(current_tokens, current_entity))
                current_entity = entity_type
                current_tokens = [clean_token]

        else:
            html_parts.append(f'<span class="token unknown">{escape_html(clean_token)}</span> ')

    if current_entity:
        html_parts.append(render_entity_span(current_tokens, current_entity))

    return ''.join(html_parts)

def render_span_text(text, entities):
    """Render span text with entity highlighting."""
    if not entities:
        return escape_html(text)

    # Sort entities by start position
    sorted_entities = sorted(entities, key=lambda x: x['start'])

    html_parts = []
    last_end = 0

    for entity in sorted_entities:
        start, end = entity['start'], entity['end']
        label = entity['label']

        # Add text before entity
        if start > last_end:
            html_parts.append(escape_html(text[last_end:start]))

        # Add entity span
        entity_text = text[start:end]
        color = get_entity_color(label)
        html_parts.append(
            f'<span class="entity" style="background-color: {color}; '
            f'padding: 2px 4px; border-radius: 3px; margin: 0 1px; '
            f'border: 1px solid #ddd;" title="{label}: {entity_text}">'
            f'{escape_html(entity_text)}</span>'
        )

        last_end = end

    # Add remaining text
    if last_end < len(text):
        html_parts.append(escape_html(text[last_end:]))

    return ''.join(html_parts)

def render_entity_span(tokens, entity_type):
    """Render entity span with styling."""
    entity_text = ' '.join(tokens)
    color = get_entity_color(entity_type)
    return (
        f'<span class="entity" style="background-color: {color}; '
        f'padding: 2px 4px; border-radius: 3px; margin: 0 1px; '
        f'border: 1px solid #ddd;" title="{entity_type}: {entity_text}">'
        f'{escape_html(entity_text)}</span> '
    )

def escape_html(text):
    """Escape HTML characters and normalize whitespace."""
    # First normalize whitespace and remove problematic characters
    text = ' '.join(text.split())  # This removes all extra whitespace including newlines

    return (text.replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#x27;'))

@app.route('/')
def index():
    """Main page showing available datasets."""
    base_dir = request.args.get('base_dir', '.')

    if not os.path.isabs(base_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(script_dir, base_dir)

    datasets = discover_datasets(base_dir)
    stats = get_dataset_statistics(datasets)

    template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Viewer</title>
    <style>
        * { box-sizing: border-box; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #1e293b;
            line-height: 1.6;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
            color: white;
            padding: 24px 32px;
            text-align: center;
        }

        .header h1 {
            margin: 0;
            font-size: 2rem;
            font-weight: 700;
        }

        .header p {
            margin: 8px 0 0 0;
            opacity: 0.9;
            font-size: 1.1rem;
        }

        .content {
            padding: 32px;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 32px;
        }

        .stat-card {
            background: #f8fafc;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #e2e8f0;
        }

        .stat-value {
            font-size: 2rem;
            font-weight: 700;
            color: #2563eb;
            margin-bottom: 4px;
        }

        .stat-label {
            color: #64748b;
            font-size: 0.9rem;
        }

        .datasets-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 24px;
        }

        .dataset-card {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            transition: all 0.2s;
        }

        .dataset-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }

        .dataset-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 8px;
        }

        .dataset-info {
            color: #64748b;
            font-size: 0.9rem;
            margin-bottom: 16px;
        }

        .view-button {
            background: #2563eb;
            color: white;
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            text-decoration: none;
            display: inline-block;
            font-size: 0.9rem;
            font-weight: 500;
            transition: background 0.2s;
        }

        .view-button:hover {
            background: #1d4ed8;
        }

        .entity-legend {
            background: #f8fafc;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 24px;
        }

        .entity-legend h3 {
            margin: 0 0 16px 0;
            color: #1e293b;
        }

        .legend-item {
            display: inline-block;
            margin: 4px 8px 4px 0;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85rem;
            font-weight: 500;
            border: 1px solid #ddd;
        }

        .empty-state {
            text-align: center;
            padding: 48px;
            color: #64748b;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Dataset Viewer</h1>
            <p>Visualization tool for voting NER datasets</p>
        </div>

        <div class="content">
            {% if stats %}
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{{ stats.total_ner_examples + stats.total_span_examples }}</div>
                    <div class="stat-label">Total Examples</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ stats.languages|length }}</div>
                    <div class="stat-label">Languages</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ stats.entity_distribution|length }}</div>
                    <div class="stat-label">Entity Types</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ datasets|length }}</div>
                    <div class="stat-label">Datasets</div>
                </div>
            </div>
            {% endif %}

            {% if stats.entity_distribution %}
            <div class="entity-legend">
                <h3>Entity Types</h3>
                {% for entity_type, count in stats.entity_distribution.most_common() %}
                <span class="legend-item" style="background-color: {{ get_entity_color(entity_type) }};">
                    {{ entity_type }} ({{ count }})
                </span>
                {% endfor %}
            </div>
            {% endif %}

            {% if datasets %}
            <div class="datasets-grid">
                {% for name, info in datasets.items() %}
                <div class="dataset-card">
                    <div class="dataset-title">{{ name }}</div>
                    <div class="dataset-info">
                        {{ info.count }} examples • {{ info.type.upper() }} format
                    </div>
                    <a href="/view?file_path={{ info.file_path }}&type={{ info.type }}" class="view-button">
                        View Dataset
                    </a>
                </div>
                {% endfor %}
            </div>
            {% else %}
            <div class="empty-state">
                <h3>No datasets found</h3>
                <p>Place your datasets in the publication_datasets directory</p>
            </div>
            {% endif %}
        </div>
    </div>
</body>
</html>
    """

    return render_template_string(
        template,
        datasets=datasets,
        stats=stats,
        get_entity_color=get_entity_color
    )

@app.route('/view')
def view_dataset():
    """View a specific dataset."""
    file_path = request.args.get('file_path')
    dataset_type = request.args.get('type')
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))

    if not file_path or not os.path.exists(file_path):
        return f"<h1>Error</h1><p>File not found: {file_path}</p>", 404

    try:
        if dataset_type == 'ner':
            examples = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        example = json.loads(line)
                        example['line_number'] = line_num
                        examples.append(example)

        elif dataset_type == 'span':
            with open(file_path, 'r', encoding='utf-8') as f:
                examples = json.load(f)
            for i, example in enumerate(examples, 1):
                example['line_number'] = i

        else:
            return "<h1>Error</h1><p>Unknown dataset type</p>", 400

        # Pagination
        total_examples = len(examples)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_examples = examples[start_idx:end_idx]

        total_pages = (total_examples + per_page - 1) // per_page
        has_prev = page > 1
        has_next = page < total_pages

        # Entity statistics
        entity_stats = Counter()
        if dataset_type == 'ner':
            for example in examples:
                for tag in example.get('labels', []):
                    if tag.startswith('B-'):
                        entity_stats[tag[2:]] += 1
        else:
            for example in examples:
                for entity in example.get('entities', []):
                    entity_stats[entity['label']] += 1

    except Exception as e:
        return f"<h1>Error</h1><p>Could not load file: {e}</p>", 500

    template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Viewer - {{ file_name }}</title>
    <style>
        * { box-sizing: border-box; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #1e293b;
            line-height: 1.6;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
            color: white;
            padding: 24px 32px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .header h1 {
            margin: 0;
            font-size: 1.5rem;
            font-weight: 600;
        }

        .back-link {
            color: white;
            text-decoration: none;
            background: rgba(255,255,255,0.2);
            padding: 8px 16px;
            border-radius: 4px;
            transition: all 0.2s;
        }

        .back-link:hover {
            background: rgba(255,255,255,0.3);
        }

        .content {
            padding: 32px;
        }

        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
            background: #f8fafc;
            padding: 20px;
            border-radius: 8px;
        }

        .info-item {
            text-align: center;
        }

        .info-value {
            font-size: 1.5rem;
            font-weight: 600;
            color: #2563eb;
        }

        .info-label {
            font-size: 0.9rem;
            color: #64748b;
        }

        .pagination {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 24px;
            gap: 8px;
        }

        .pagination a, .pagination span {
            padding: 8px 12px;
            border: 1px solid #e2e8f0;
            border-radius: 4px;
            text-decoration: none;
            color: #1e293b;
            background: white;
        }

        .pagination a:hover {
            background: #2563eb;
            color: white;
        }

        .pagination .current {
            background: #2563eb;
            color: white;
        }

        .example-card {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        .example-header {
            background: #f8fafc;
            padding: 12px 20px;
            border-bottom: 1px solid #e2e8f0;
            font-weight: 600;
            color: #1e293b;
        }

        .example-content {
            padding: 20px;
            line-height: 1.8;
            font-size: 1rem;
            word-wrap: break-word;
        }

        .token {
            margin-right: 4px;
        }

        .entity {
            cursor: help;
            transition: all 0.2s;
            font-weight: 500;
        }

        .entity:hover {
            opacity: 0.8;
            transform: scale(1.02);
        }

        .error {
            background: #ef4444;
            color: white;
            padding: 12px;
            border-radius: 4px;
            margin: 8px 0;
        }

        .unknown {
            background: #f59e0b;
            color: white;
            padding: 2px 4px;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📄 {{ file_name }} ({{ dataset_type.upper() }})</h1>
            <a href="/" class="back-link">← Back</a>
        </div>

        <div class="content">
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-value">{{ total_examples }}</div>
                    <div class="info-label">Examples</div>
                </div>
                <div class="info-item">
                    <div class="info-value">{{ page }}</div>
                    <div class="info-label">Page</div>
                </div>
                <div class="info-item">
                    <div class="info-value">{{ total_pages }}</div>
                    <div class="info-label">Total Pages</div>
                </div>
                <div class="info-item">
                    <div class="info-value">{{ entity_stats|length }}</div>
                    <div class="info-label">Entity Types</div>
                </div>
            </div>

            <div class="pagination">
                {% if has_prev %}
                <a href="?file_path={{ file_path }}&type={{ dataset_type }}&page={{ page - 1 }}&per_page={{ per_page }}">← Previous</a>
                {% endif %}

                <span class="current">{{ page }} / {{ total_pages }}</span>

                {% if has_next %}
                <a href="?file_path={{ file_path }}&type={{ dataset_type }}&page={{ page + 1 }}&per_page={{ per_page }}">Next →</a>
                {% endif %}
            </div>

            {% for example in page_examples %}
            <div class="example-card">
                <div class="example-header">
                    {% if example.get('id') %}
                        {{ example.id }}
                    {% elif example.get('segment_id') %}
                        {{ example.segment_id }}
                    {% else %}
                        Example {{ example.line_number }}
                    {% endif %}

                    {% if example.get('municipality_id') %}
                        • {{ example.municipality_id }}
                    {% endif %}
                </div>

                <div class="example-content">
                    {% if dataset_type == 'ner' %}
                        {{ render_ner_tokens(example.get('tokens', []), example.get('labels', [])) | safe }}
                    {% elif dataset_type == 'span' %}
                        {{ render_span_text(example.get('text', ''), example.get('entities', [])) | safe }}

                        {% if example.get('canonical_target') %}
                        <div style="margin-top: 16px; padding-top: 16px; border-top: 1px solid #e2e8f0; font-size: 0.9rem; color: #64748b;">
                            <strong>Canonical targets:</strong>
                            {% for label, values in example.canonical_target.items() %}
                                <div>{{ label }}: {{ values | join(', ') }}</div>
                            {% endfor %}
                        </div>
                        {% endif %}
                    {% endif %}
                </div>
            </div>
            {% endfor %}

            <div class="pagination">
                {% if has_prev %}
                <a href="?file_path={{ file_path }}&type={{ dataset_type }}&page={{ page - 1 }}&per_page={{ per_page }}">← Previous</a>
                {% endif %}

                <span class="current">{{ page }} / {{ total_pages }}</span>

                {% if has_next %}
                <a href="?file_path={{ file_path }}&type={{ dataset_type }}&page={{ page + 1 }}&per_page={{ per_page }}">Next →</a>
                {% endif %}
            </div>
        </div>
    </div>
</body>
</html>
    """

    file_name = os.path.basename(file_path)

    return render_template_string(
        template,
        file_name=file_name,
        file_path=file_path,
        dataset_type=dataset_type,
        page_examples=page_examples,
        total_examples=total_examples,
        page=page,
        total_pages=total_pages,
        has_prev=has_prev,
        has_next=has_next,
        per_page=per_page,
        entity_stats=entity_stats,
        render_ner_tokens=render_ner_tokens,
        render_span_text=render_span_text
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Viewer for Publication Datasets')
    parser.add_argument('--port', type=int, default=5000, help='Port to run server on')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')

    args = parser.parse_args()

    print(f"Starting Dataset Viewer on http://{args.host}:{args.port}")
    print("Point your browser to view the publication datasets")

    app.run(host=args.host, port=args.port, debug=args.debug)