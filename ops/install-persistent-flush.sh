set -u
chmod +x ~/persistent-flush.sh
systemctl is-active cron >/dev/null 2>&1 || { echo "$(hostname): cron service NOT active -- skipped"; exit 1; }
sudo -k
if ~/persistent-flush.sh; then r=$(tail -1 /var/tmp/persistent-flush.log); else echo "$(hostname): TEST FAILED -> $(tail -1 /var/tmp/persistent-flush.log)"; exit 1; fi
( crontab -l 2>/dev/null | grep -v 'persistent-flush.sh'; echo '*/5 * * * * $HOME/persistent-flush.sh' ) | crontab -
echo "$(hostname): test [$r] cron entries=$(crontab -l 2>/dev/null | grep -c persistent-flush.sh)"
