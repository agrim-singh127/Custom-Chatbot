# Introduction 
Advanced custom chatbot for specific reports/dashboards. It can be launched into frontend of websites and also in power bi reports via custom visuals. It is superior and economic to what copilot can provide, especially when data is large, complex where security and rls restrictions are the priority.
If you find this project helpful, feel free to ⭐ star it or 🍴 fork it. Your support is appreciated and helps the project grow!

# Getting Started
Setting the code up and running requires some effort:
1.	Download the files & start by creating a virtual environment, use these lines in your cmd terminal: 
        python -m venv demo
        demo\Scripts\activate
        pip install -required.txt
    With this you will have necessary free tools with you. Now onto main part
2.	Current setup is a customized verion that encompasses rls based on mailing, based on requirements comment out not required sections
3.	open ai/llm used here for texttosql query and response is paid, you can change api url to free versions from other services such as hugging face, etc and get your api token
4.	SQL server is used for database storage, keeping in mind, chatbot will be used across the organization, for runnig locally you can use free sqlite as well
5.  Username, passwords, tokens are all replaced with dummy data, replace those sections
6.  Enter your database tables in metadata file, prompts in demo file for it to function properly
7.  For local testing use fastapi but from server to local you need to create a internet facing DNS which is not free
8.  For chatbot outcome you need to write frontend code if for websites, custom visual if for power bi, etc. Refer my other repositories for these

# Contribute
The current chatbot handles mostly data descriptively, for prescriptive insights as well such as forecasting better models needs to be created, no change here but a custom llm itself.
Happy to collaborate with such as expert to push it next level.

# Licensing
Need to notify owner, purpose for using. send mail at: agrimkhiladigref@gmail.com

# Page Views
![Visitors](https://visitor-badge.laobi.icu/badge?page_id=agrim-singh127.Custom-Chatbot) 
