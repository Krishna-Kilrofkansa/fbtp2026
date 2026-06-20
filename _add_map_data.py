"""Add mapEvents to data.js for the React dashboard."""
import json

with open("_react_events.json") as f:
    events = json.load(f)

causes = sorted(set(e["cause"] for e in events))

with open("react-dashboard/src/data.js", "r", encoding="utf-8") as f:
    content = f.read()

# Build the map data block
lines = []
lines.append("  // Sample events for map (300 representative points)")
lines.append("  mapEvents: [")
for e in events:
    lat = e["lat"]
    lng = e["lng"]
    cause = e["cause"]
    priority = e["priority"]
    impact = e["impact"]
    corridor = e["corridor"]
    lines.append(f'    {{lat:{lat},lng:{lng},cause:"{cause}",priority:"{priority}",impact:{impact},corridor:"{corridor}"}},')
lines.append("  ],")
lines.append("")
lines.append(f"  allCauses: {json.dumps(causes)},")

map_block = "\n".join(lines)

# Insert before the closing brace + newline
# Find the last "}" in the file
idx = content.rfind("}")
if idx > 0:
    content = content[:idx] + "\n" + map_block + "\n}\n"

with open("react-dashboard/src/data.js", "w", encoding="utf-8") as f:
    f.write(content)

print(f"Added {len(events)} map events and {len(causes)} causes to data.js")
