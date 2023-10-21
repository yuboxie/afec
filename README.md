# AFEC: A Knowledge Graph Capturing Social Intelligence in Casual Conversations

The filtering process contains three steps:
1. Preprocessing
2. Summarization
3. Dependency parsing

Description:
- `scripts`: Code for building the knowledge graph
    1. Crawl from Reddit
    2. Filter the submissions and comments of Reddit
    3. Filter ED, and then merge with Reddit
    4. Cluster
- `data`: All the data for building the knowledge graph
    - `ed`: First and second turns of ED after filtering
    - `reddit`: Reddit submissions and comments
        - `raw`: Raw data crawled using PushShift APIs
        - `filtered`: Submissions and comments after filtering
        - `matched`: Submissions and comments matched into single `csv` files
    - `merged`: Reddit and ED merged
- `plot`: Some visualizations

Due to the file size limitation of GitHub, some files are stored on [Google Drive](https://drive.google.com/drive/folders/1elcF3KnVKDCz-fW2rBVVZL-DUCgA6-3P?usp=sharing):
- `data/merged/all_events_embed.npy`
- `data/merged/all_replies_embed.npy`
- `data/reddit/raw/*`
