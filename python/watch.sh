rsync -av . ml-schlesinger:/data/nicow/qblade/python --exclude 'logs' --exclude 'checkpoints' 
while inotifywait -r -e close_write,modify,move,attrib,delete,create . 
do
	echo "updating"
 	rsync -av . ml-schlesinger:/data/nicow/qblade/python --exclude 'logs' --exclude 'checkpoints' 
done