import pymysql
#cities, phrases, memo

conn = pymysql.connect(host='lenbur.online',
                             user='admww_fox',
                             password='PassWord...',
                             database='admww_fox',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)
cursor = conn.cursor()
 
def sql(code):
	cursor.execute(code)
	
#sql('''

#Сюда можно писать sql код


#''')


def create(tab_name, columns, conn = conn):
	with conn:
		columns = str(columns)
		columns = columns.replace("'", "")
		code = 'CREATE TABLE ' + str(tab_name) + ' (' + columns + ')'
		sql(code)
	return code

	#удаление таблицы
def drop(tab_name, conn = conn):
	with conn:
		code = 'DROP TABLE ' + str(tab_name)
		sql(code)
	return code

	#занесение данных
def dbinsert(tab_name, columns, conn = conn):
	with conn:
		columns = columns.replace("'", "")
		columns = columns.replace('"', "'")
		code = 'INSERT INTO ' + tab_name + " VALUES("+ columns + ")"
		sql(code)
		conn.commit()
	return columns

	#извлечение данных
def dboutput(tab_name, column='*', optcom="", conn = conn):
	with conn:
		code=("SELECT " + column + " FROM " + tab_name + optcom)
		sql(code)
	return cursor.fetchall()

def tablist(conn=conn):
	with conn:
		code=("select * from sqlite_master where type = 'table'")
		sql(code)
		lis = cursor.fetchall()
		lisa = []
		res = ''
		for i in range(len(lis) - 1):
			res += (lis[i][1] + ', ')
		res += lis[-1][1]
	return res

def piddelete(pid,conn = conn):
	with conn:
		code = "DELETE FROM pslist WHERE PID = '" + str(pid) + "'"
		print(code)
		sql(code)
	return code
def namedelete(name,conn = conn):
	with conn:
		code = "DELETE FROM pslist WHERE Name = '" + str(name) + "'"
		print(code)
		sql(code)
	return code

