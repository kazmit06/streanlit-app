import streamlit as st

def main():
    home = st.Page(
        page="contents/home.py", title="ホーム", icon="🏠", default=True
    )
    analysis = st.Page(
        page="contents/app.py", title="データ分析", icon="📊"
    )
    settings = st.Page(
        page="contents/settings_page.py", title="設定", icon="⚙️"
    )

    pg = st.navigation([home, analysis, settings])
    pg.run()

if __name__ == "__main__":
    main()
