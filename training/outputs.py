"""Portable, offline reports and safe artifact writes (no new dependencies)."""
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path
import tempfile
import uuid
import numpy as np
from PIL import Image, ImageDraw
from common import write_json, sha256


@contextmanager
def evaluation_attempt(filename):
    """Keep a failed new attempt distinguishable from an older successful JSON."""
    import traceback
    dest=Path(filename); marker=Path(str(dest)+'.attempt.json')
    attempt={'started_utc':datetime.now(timezone.utc).isoformat(), 'status':'running',
        'output':str(dest), 'previous_result_sha256':sha256(dest) if dest.is_file() else None}
    write_json(marker,attempt)
    try:
        yield
        write_json(marker,dict(attempt,status='complete',result_sha256=sha256(dest)))
    except BaseException as exc:
        write_json(marker,dict(attempt,status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),
            note='Previous successful result, if present, was not replaced by this failed attempt.'))
        raise


def fresh_directory(output):
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory is not empty; choose a NEW directory: '+str(out))
    out.mkdir(parents=True, exist_ok=True)
    return out


def csv_cell(value):
    if isinstance(value, (dict, list)): value = json.dumps(value, ensure_ascii=False)
    # Identifiers/paths originate in input data. Prevent spreadsheet formula execution.
    if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@')): return "'"+value
    return value


def write_csv(filename, fields, rows):
    dest = Path(filename); dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix+'.partial')
    try:
        with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            for row in rows: writer.writerow({k: csv_cell(v) for k, v in row.items()})
        os.replace(temporary, dest)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_export(filename, payload):
    """Reload/compare before replacing a previously valid inference artifact."""
    import torch
    dest = Path(filename); dest.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=dest.name+'.', suffix='.partial', dir=dest.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        loaded = torch.load(temporary, map_location='cpu', weights_only=True)
        if loaded.keys() != payload.keys() or loaded['state_dict'].keys() != payload['state_dict'].keys():
            raise ValueError('Export schema reload mismatch')
        if not all(torch.equal(v, loaded['state_dict'][k]) for k, v in payload['state_dict'].items()):
            raise ValueError('Export tensor reload mismatch')
        os.replace(temporary, dest)
    finally:
        Path(temporary).unlink(missing_ok=True)


def table(headers, rows):
    esc = lambda value: html.escape(str(value))
    return '<table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'


def page(title, content):
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>'+html.escape(title)+'</title><style>body{font:16px system-ui,sans-serif;max-width:1400px;margin:30px auto;padding:0 20px;line-height:1.6}table{border-collapse:collapse;display:block;overflow:auto}td,th{padding:8px;border:1px solid #ccc;text-align:left}th{background:#edf2f7}img{max-width:100%}.warning{background:#fff0cf;padding:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>'+html.escape(title)+'</h1>'+content+'</html>'


def write_evaluation(filename, report, manifest, render_images=False):
    """Commit JSON last; each rendering has a distinct directory, so no stale images."""
    if report['manifest_id'] != manifest['manifest_id']: raise ValueError('Report/manifest mismatch')
    groups = {g['id']: g for g in manifest['groups'] if g['split'] == report['split']}
    if not groups: raise ValueError('Empty evaluation split')
    if not report['results']: raise ValueError('No evaluation methods')
    query_rows, metric_rows = [], []
    for mode, result in report['results'].items():
        rows = result['rows']
        if len(rows) != len(groups) or {r['id'] for r in rows} != set(groups):
            raise ValueError('Incomplete/duplicate evaluation rows for '+mode)
        for row in rows:
            g = groups[row['id']]; count = len(g['candidates'])
            if row['candidate_count'] != count or sorted(row['order']) != list(range(count)):
                raise ValueError('Invalid candidate order: '+row['id'])
            scores = row.get('scores')
            if scores is not None and (np.asarray(scores).shape != (count,) or not np.isfinite(scores).all()):
                raise ValueError('Invalid candidate scores: '+row['id'])
            rank = row['first_positive_rank']
            expected_rank = next((i+1 for i,j in enumerate(row['order']) if g['candidates'][j]['label']==1), None)
            if rank != expected_rank or row['covered'] != (expected_rank is not None):
                raise ValueError('Rank/coverage disagrees with candidate labels: '+row['id'])
            if scores is not None and row['order'] != np.argsort(-np.asarray(scores), kind='stable').tolist():
                raise ValueError('Ranking disagrees with scores: '+row['id'])
            outcome = 'no_positive_candidate' if not row['covered'] else ('top1_hit' if rank == 1 else 'retrieval_miss_top1')
            query_rows.append({'mode': mode, 'id': row['id'], 'group': g['group'], 'sequence': g['sequence'],
                'protocol': g['protocol'], 'candidate_count': count, 'covered': row['covered'],
                'first_positive_rank': rank, 'outcome': outcome,
                'top_candidate_index': row['order'][0] if count else None,
                'top_score': scores[row['order'][0]] if count and scores is not None else None})
        for bucket, metrics in [('overall', result['overall']), *result['groups'].items()]:
            metric_rows.append({'mode': mode, 'bucket': bucket, 'queries': metrics['queries'],
                'covered_queries': metrics['covered_queries'], 'candidate_coverage': metrics['candidate_coverage'],
                **{'recall_all_at_'+k: v for k, v in metrics['recall_all_legal'].items()},
                **{'recall_covered_at_'+k: v for k, v in metrics['recall_conditional_covered'].items()}})
    dest = Path(filename); run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    out = fresh_directory(dest.parent/(dest.stem+'_reports')/run)
    write_json(out/'status.json', {'status': 'building', 'manifest_id': manifest['manifest_id']})
    try:
        write_csv(out/'queries.csv', list(query_rows[0]), query_rows)
        write_csv(out/'metrics.csv', list(metric_rows[0]), metric_rows)
        def candidates():
            for mode, result in report['results'].items():
                for row in result['rows']:
                    for rank, index in enumerate(row['order'], 1):
                        c = groups[row['id']]['candidates'][index]
                        yield {'mode': mode, 'id': row['id'], 'rank': rank, 'candidate_index': index,
                            'score': row['scores'][index] if 'scores' in row else None,
                            'label': c['label'], 'coverage': c['coverage'], 'scale_m': c['scale_m'],
                            'valid_fraction': c['valid_fraction'], 'center_map': c['center_map']}
        write_csv(out/'candidates.csv', ['mode','id','rank','candidate_index','score','label','coverage','scale_m','valid_fraction','center_map'], candidates())
        warning = 'SYNTHETIC ONLY — 合成数据，仅验证程序；不代表真实精度。' if report['synthetic'] else '指标依赖当前几何生成的候选标签；不能替代独立真实地标核验。'
        content = '<p class="warning">'+warning+'</p><p>split: '+html.escape(report['split'])+' · manifest: '+html.escape(report['manifest_id'])+'</p>'
        content += '<p>无正候选计入总体召回失败；有覆盖但排名靠后单列为检索失败。模型分数是相似度，不是置信概率。</p>'
        content += '<p><a href="queries.csv">逐查询 CSV</a> · <a href="metrics.csv">分组指标 CSV</a> · <a href="candidates.csv">完整候选排名 CSV</a> · <a href="evaluation.json">完整 JSON</a></p>'
        content += table(['方案','查询数','有正候选','覆盖率','R@1 全部','R@5 全部','R@1 有覆盖'], [
            [r['mode'],r['queries'],r['covered_queries'],r['candidate_coverage'],r['recall_all_at_1'],r['recall_all_at_5'],r['recall_covered_at_1']]
            for r in metric_rows if r['bucket']=='overall'])
        worst = sorted(query_rows, key=lambda r: (r['outcome']=='top1_hit', -(r['first_positive_rank'] or 10**9), r['id']))[:100]
        content += '<h2>失败优先的查询明细（最多 100 行；完整内容在 CSV）</h2>'+table(
            ['方案','查询','结果','首正样本排名','候选数'], [[r['mode'],r['id'],r['outcome'],r['first_positive_rank'],r['candidate_count']] for r in worst])
        content += '<h2>数据处理范围</h2><pre>'+html.escape(json.dumps(report.get('data_flow', {}), ensure_ascii=False, indent=2))+'</pre>'
        visualizations = {}
        if render_images:
            from diagnostics import contact_sheet
            content += '<h2>候选图（每组选择本方案最差查询；跨方案可能不同）</h2>'
            for number, mode in enumerate(report['results']):
                name=f'contact_sheet_{number:02}.png'
                try:
                    visualizations[mode]=contact_sheet(manifest, out/name, report, mode)
                    content += '<h3>'+html.escape(mode)+'</h3><p>'+html.escape(visualizations[mode]['status'])+'</p><a href="'+name+'"><img src="'+name+'"></a>'
                except (ValueError, OSError) as exc:
                    visualizations[mode]={'status':'failed','errors':[str(exc)]}
                if visualizations[mode].get('errors'):
                    content += '<pre>'+html.escape(json.dumps(visualizations[mode]['errors'],ensure_ascii=False,indent=2))+'</pre>'
        (out/'index.html').write_text(page('检索评估报告', content), encoding='utf-8')
        result = dict(report, outputs={'report_directory': out.relative_to(dest.parent).as_posix(), 'html': (out/'index.html').relative_to(dest.parent).as_posix(), 'visualizations':visualizations})
        write_json(out/'evaluation.json', result)
        files = [f.name for f in out.iterdir() if f.name!='status.json']
        status='complete_with_visualization_errors' if any(v['status']!='complete' for v in visualizations.values()) else 'complete'
        write_json(out/'status.json', {'status': status, 'files_sha256': {f: sha256(out/f) for f in files}})
        write_json(dest, result)  # Latest pointer/data changes only after every sidecar succeeded.
        return out
    except Exception as exc:
        write_json(out/'status.json', {'status': 'failed', 'error': str(exc)})
        raise


def write_training_history(output, history):
    out = Path(output)
    write_json(out/'history.json', history)
    if not history: return
    fields = list(dict.fromkeys(k for row in history for k in row))
    write_csv(out/'history.csv', fields, history)
    canvas = Image.new('RGB', (1100, 660), '#101924'); draw = ImageDraw.Draw(canvas)
    draw.text((30, 16), 'TRAINING TRACE - validation selection only; epoch numbers are zero-based', fill='white')
    for panel, key in enumerate(('loss', 'val_recall_at_1_all')):
        x0, y0, w, h = 85, 70+panel*300, 940, 220
        values = [(row['epoch'], row.get(key)) for row in history]
        finite = [(x,y) for x,y in values if isinstance(y,(int,float)) and math.isfinite(y)]
        draw.text((x0,y0-23), key, fill='white'); draw.rectangle((x0,y0,x0+w,y0+h), outline='#7e93a9')
        if not finite:
            draw.text((x0+15,y0+20), 'No finite values', fill='#ffc06e'); continue
        low,high = (0,1) if 'recall' in key else (min(y for x,y in finite),max(y for x,y in finite))
        if high==low: low,high=low-.05,high+.05
        xmin,xmax=min(x for x,_ in values),max(x for x,_ in values)
        draw.text((10,y0),f'{high:.4g}',fill='white'); draw.text((10,y0+h-12),f'{low:.4g}',fill='white')
        draw.text((x0,y0+h+8),str(xmin),fill='white'); draw.text((x0+w-20,y0+h+8),str(xmax),fill='white')
        previous=None
        for x,y in values:
            if not isinstance(y,(int,float)) or not math.isfinite(y): previous=None; continue
            point=(x0+(x-xmin)/max(1,xmax-xmin)*w, y0+h-(y-low)/(high-low)*h)
            if previous: draw.line([previous,point],fill='#61d7e8',width=3)
            draw.ellipse((point[0]-3,point[1]-3,point[0]+3,point[1]+3),fill='#ffd46d'); previous=point
    temp=out/'history.png.partial'; canvas.save(temp, format='PNG'); os.replace(temp,out/'history.png')


if __name__ == '__main__':
    import argparse
    from common import read_json
    p=argparse.ArgumentParser(description='Render existing MLP/LoRA evaluation or training history without GPU')
    p.add_argument('--evaluation'); p.add_argument('--manifest'); p.add_argument('--output')
    p.add_argument('--history'); a=p.parse_args()
    if a.history: write_training_history(a.output or Path(a.history).parent, read_json(a.history))
    elif a.evaluation and a.manifest and a.output:
        print(write_evaluation(a.output, read_json(a.evaluation), read_json(a.manifest), render_images=True)/'index.html')
    else: p.error('Use --history [--output DIR] or --evaluation FILE --manifest FILE --output JSON')
