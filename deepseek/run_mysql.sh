docker run -d \
  --name data-synthesis-database \
  -p 3306:3306 \
  -e MYSQL_ROOT_PASSWORD=rootpassword \
  -e MYSQL_DATABASE=datasynthesis \
  -e MYSQL_USER=myuser \
  -e MYSQL_PASSWORD=mypassword \
  -v /home/kmanasu/inference-experiment/deepseek/data:/var/lib/mysql \
  mysql:8

# docker exec -it data-synthesis-database mysql -u root -p

# CREATE DATABASE mockdatadb;
# USE mockdatadb;
# CREATE TABLE conversations (id INT AUTO_INCREMENT PRIMARY KEY, kb VARCHAR(255), conversation VARCHAR(2048));
# SHOW TABLES;
