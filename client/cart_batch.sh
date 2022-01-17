#!/bin/bash
# gateway: ("a32104293011" "a32104293013" "a32104293016")
cart='a32104293011'

# for cart in "${carts[@]}"
# do
#     echo "Select Gateway: "$cart
#     python3 /home/p2g/bellk/fuelcell/client/cart_client.py $cart
# done

echo "Select Gateway: "$cart
python3 /home/p2g/bellk/fuelcell/client/cart_client.py $cart
