import os
import runpy

# Ensure a dummy OPENAI_API_KEY so app.py doesn't st.stop()
os.environ.setdefault('OPENAI_API_KEY', 'test-placeholder')

# Use the script's directory to refer to app.py reliably
script_dir = os.path.dirname(os.path.abspath(__file__))
app_path = os.path.join(script_dir, 'app.py')

# Run app.py with its directory as CWD so relative data paths resolve
os.chdir(script_dir)
ns = runpy.run_path(app_path)

# Basic sanity checks
print('MODULE_KEYS:', sorted(k for k in ns.keys() if not k.startswith('__'))[:40])

# Test build_gpt_context with small slices
all_nodes = ns.get('all_nodes', [])
historical_recipes = ns.get('historical_recipes', [])

nodes_sample = all_nodes[:20]
recipes_sample = historical_recipes[:10]

if 'build_gpt_context' in ns:
    nodes_ctx, recipes_ctx = ns['build_gpt_context'](nodes_sample, recipes_sample)
    print('build_gpt_context ->', len(nodes_ctx), 'nodes_ctx,', len(recipes_ctx), 'recipes_ctx')
else:
    print('build_gpt_context not found')

if 'create_network_graph' in ns and nodes_sample:
    center = nodes_sample[0].get('Label', 'CENTER')
    connected = [{'name': nodes_sample[i].get('Label', f'N{i}'), 'degree': int(nodes_sample[i].get('Degree', 1) or 1), 'type': nodes_sample[i].get('node_type', 'unknown')} for i in range(1, min(4, len(nodes_sample)))]
    fig = ns['create_network_graph'](center, connected)
    print('create_network_graph ->', type(fig))
else:
    print('create_network_graph not run')

# Fasting recipe check
if 'is_fasting_recipe' in ns and historical_recipes:
    count = sum(1 for r in historical_recipes if ns['is_fasting_recipe'](r))
    print('fasting recipes found:', count)
else:
    print('is_fasting_recipe not checked')

print('SMOKE TEST DONE')
