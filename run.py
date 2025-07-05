import uvicorn

if __name__ == "__main__":
    uvicorn.run('demo:app',host='127.0.0.1',port=8044,log_level='debug',reload=True)