make watch
while inotifywait -r -e close_write,modify,move,attrib,delete,create . 
do
	echo "updating"
 	make watch
done