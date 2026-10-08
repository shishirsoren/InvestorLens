# Publish InvestorLens so anyone can access it

## Option A — Streamlit Community Cloud
1. Create a GitHub repository and upload the contents of this folder.
2. In Streamlit Community Cloud, create a new app.
3. Select the repository and set the main file to `app.py`.
4. Deploy.
5. Add `FINNHUB_API_KEY` under the app's Secrets if you have one. Never commit API keys.
6. Share the generated `streamlit.app` URL. Anyone with the URL can open the app.

## Option B — Render
1. Push this folder to GitHub.
2. In Render, create a new Web Service from the repository.
3. Render will detect `render.yaml` / `Dockerfile`.
4. Deploy and share the generated `onrender.com` URL.
5. Add API keys as environment variables in Render, never in Git.

## Important
A local Streamlit URL such as `localhost:8501` is only accessible from the computer running the app. Public access requires a hosted deployment.

For a production investment product, use a licensed market-data/fundamentals provider for exchange-grade real-time data and verify financial statement values against primary filings.
