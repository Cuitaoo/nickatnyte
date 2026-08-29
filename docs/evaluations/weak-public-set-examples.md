# Weak Public-Set Examples - Enabled Vector + Cross-Encoder

Generated from the current `0.851142` branch with `.env` loaded: `TECHJAM_VECTOR_ENABLED=true`, `TECHJAM_RERANK_ENABLED=true`, `OPENAI_ENABLED=false`.

Current enabled run summary:

```text
score=0.851142
hit@10=0.980
mrr=0.705141
mttc=3.520
```

These are the worst sessions by session-level scoring loss: misses first, then late hits or low-rank hits.

## 1. public_0144 - intent_override - miss

Target: `B08LMMDYV7 | URBAN REPUBLIC Women's Winter Jacket - Heavyweight Water Resistant Expedition Faux-Fur Lined... | Clothing, Shoes & Jewelry > Women > Clothing > Coats, Jackets & Vests > Down ...`

Intent constraints:

```text
hard: polyester; 100% Polyester
soft: Imported; Zipper closure
```

Evaluator result: hit=False, first_hit_turn=None, rank=None, reciprocal_rank=0.000000

Likely problem: Recall miss: even after available clarifications, the target never reached the scored top 10. Generic metadata or late open-ended clarification is part of the pattern.

Conversation:

```text
T1 user: I'm looking for Jackets & Vests Down Jackets & Parkas. Zipper closure
   ask_attribute=material, target not in returned list
T2 user: For that, what matters is: polyester; 100% Polyester.
   ask_attribute=color, target not in returned list
T3 user: I don't have an additional preference for color.
   ask_attribute=feature, target rank 8
T4 user: Actually, ignore my earlier preference. What I need is: polyester.
   ask_attribute=brand, target not in returned list
T5 user: I don't have an additional preference for brand.
   ask_attribute=use_case, target not in returned list
T6 user: I don't have an additional preference for use_case.
   ask_attribute=size, target not in returned list
T7 user: I don't have an additional preference for size.
   ask_attribute=style, target not in returned list
T8 user: I don't have an additional preference for style.
   ask_attribute=other, target not in returned list
T9 user: For that, what matters is: Imported; Zipper closure.
   ask_attribute=None, target not in returned list
T10 user: Those options are not quite right yet. Ask me about one specific attribute.
   ask_attribute=None, target not in returned list
```

Returned list on stopping turn:

```text
1. B07Y8YH7GN | Fashiontasy Women's Short Puffer Coat with Removable Faux Fur Trim Hood Jacket | Clothing, Shoes & Jewelry > Women > Clothing > Coats, Jackets & Vests > Down ...
2. B08SS3Y79W | Amazon Essentials Women's Heavyweight Diamond Quilted Knee Length Puffer Coat | Clothing, Shoes & Jewelry > Women > Clothing > Coats, Jackets & Vests > Down ...
3. B076TTCM7S | Columbia Women's Breitenbush™ Parka Outerwear, -black, S | Clothing, Shoes & Jewelry > Women > Clothing > Coats, Jackets & Vests > Down ...
4. B09JX1Q6RY | 32 Degrees Women’s Down Alternative Water-Repellent Jacket, Lightweight Full-Zip Packable | Clothing, Shoes & Jewelry > Women > Clothing > Coats, Jackets & Vests > Down ...
5. B0995HJKRX | Womens Hooded Down Jacket Long Quilted Lightweight Puffer Coat, Standard and Plus Size | Clothing, Shoes & Jewelry > Women > Clothing > Coats, Jackets & Vests > Down ...
6. B00LEOX31O | Columbia Women's Plus Size Powder Pillow Jacket | Clothing, Shoes & Jewelry > Women > Clothing > Coats, Jackets & Vests > Down ...
7. B09D112TGF | Steve Madden Women's Puffer | Clothing, Shoes & Jewelry > Women > Clothing > Coats, Jackets & Vests > Down ...
8. B0BCK59834 | ZIAI Winter Coats For Women Thickened Short & Long Puffer Jacket Womens with Fur S-XXL | Clothing, Shoes & Jewelry > Women > Clothing > Coats, Jackets & Vests > Down ...
9. B07CZGMNML | LONDON FOG Women's Plus Size Long Mitred Quilt Down with Faux Fur Trim Hd | Clothing, Shoes & Jewelry > Women > Clothing > Coats, Jackets & Vests > Down ...
10. B07BR2J71F | U.S. Polo Assn. Women's Quilted Moto Puffer Jacket | Clothing, Shoes & Jewelry > Women > Clothing > Coats, Jackets & Vests > Down ...
```

## 2. public_0154 - buying - miss

Target: `B00CYNKSTE | Bestform Women's Wire Free Bra | Clothing, Shoes & Jewelry > Women > Clothing > Lingerie, Sleep & Lounge > Lin...`

Intent constraints:

```text
hard: cotton; color: white
soft: Hand Wash Only; Item model number: 5006715
```

Evaluator result: hit=False, first_hit_turn=None, rank=None, reciprocal_rank=0.000000

Likely problem: Recall miss: even after available clarifications, the target never reached the scored top 10. Generic metadata or late open-ended clarification is part of the pattern.

Conversation:

```text
T1 user: I'm looking for Bras Everyday Bras. A key requirement is: cotton.
   ask_attribute=color, target not in returned list
T2 user: For that, what matters is: color: white.
   ask_attribute=feature, target not in returned list
T3 user: For that, what matters is: Hand Wash Only; Item model number: 5006715.
   ask_attribute=brand, target not in returned list
T4 user: I don't have an additional preference for brand.
   ask_attribute=size, target not in returned list
T5 user: I don't have an additional preference for size.
   ask_attribute=use_case, target not in returned list
T6 user: I don't have an additional preference for use_case.
   ask_attribute=style, target not in returned list
T7 user: I don't have an additional preference for style.
   ask_attribute=other, target not in returned list
T8 user: I don't have an additional preference for other.
   ask_attribute=None, target not in returned list
T9 user: Those options are not quite right yet. Ask me about one specific attribute.
   ask_attribute=None, target not in returned list
T10 user: Those options are not quite right yet. Ask me about one specific attribute.
   ask_attribute=None, target not in returned list
```

Returned list on stopping turn:

```text
1. B0023ZZAXW | Hanes 100% Cotton Lightly Lined Soft Cup 2-Pack, 34A-White/White | Clothing, Shoes & Jewelry > Women > Clothing > Lingerie, Sleep & Lounge > Lin...
2. B000249WB2 | Va Bien Women's Lace Front Closure Bustier with Garters 810 38B White | Clothing, Shoes & Jewelry > Women > Clothing > Lingerie, Sleep & Lounge > Lin...
3. B071R1YM3N | Flower Bali Underwire Bra 2 Pack_White_40D | Clothing, Shoes & Jewelry > Women > Clothing > Lingerie, Sleep & Lounge > Lin...
4. B07F79CVQK | Eberjey Pima Cotton Women's Everyday T-Shirt Bra | Clothing, Shoes & Jewelry > Women > Clothing > Lingerie, Sleep & Lounge > Lin...
5. B098NRHGMZ | Womens Bra, Full Figure Magic Lift Plus Size Seamless Wirefree Front Close Sports Bra, Padde... | Clothing, Shoes & Jewelry > Women > Clothing > Lingerie, Sleep & Lounge > Lin...
6. B09WCYC17X | 3 Pieces Men's A Shirts Classic Men Tank Tops Cotton Tank Tops Undershirts Sleeveless Workou... | Clothing, Shoes & Jewelry > Men > Clothing > Underwear > Undershirts
7. B085L58P9F | SPECIALMAGIC Women's Sweatpants Capri Pants Cropped Jogger Running Pants Lounge Loose Fit Dr... | Clothing, Shoes & Jewelry > Women > Clothing > Active > Active Pants > Sweatp...
8. B00GDGAIS4 | NDK New York Women's Chenille Full Length Robe 100% Cotton | Clothing, Shoes & Jewelry > Women > Clothing > Lingerie, Sleep & Lounge > Sle...
9. B073XHKSXP | PRETTODAY Women's Sexy Off Shoulder Crop Tops Short Sleeve Shirts Casual Slim Tees | Clothing, Shoes & Jewelry > Women > Clothing > Tops, Tees & Blouses > T-Shirts
10. B07V36V8B6 | JerrisApparel Flower Girls Rainbow Tutu Dress Unicorn Birthday Party Skirt Outfit | Clothing, Shoes & Jewelry > Costumes & Accessories > Kids & Baby > Girls > Co...
```

## 3. public_0174 - buying - miss

Target: `B0794VPVBH | NY Threads Luxurious Mens Shawl Collar Fleece Bathrobe Spa Robe | Clothing, Shoes & Jewelry > Men > Clothing > Sleep & Lounge > Robes`

Intent constraints:

```text
hard: polyester; 100% Polyester
soft: Imported; Tie closure
```

Evaluator result: hit=False, first_hit_turn=None, rank=None, reciprocal_rank=0.000000

Likely problem: Recall miss: even after available clarifications, the target never reached the scored top 10. Generic metadata or late open-ended clarification is part of the pattern.

Conversation:

```text
T1 user: I'm looking for Sleep & Lounge Robes. A key requirement is: polyester.
   ask_attribute=feature, target not in returned list
T2 user: For that, what matters is: Imported; Tie closure.
   ask_attribute=color, target not in returned list
T3 user: I don't have an additional preference for color.
   ask_attribute=brand, target not in returned list
T4 user: I don't have an additional preference for brand.
   ask_attribute=size, target not in returned list
T5 user: I don't have an additional preference for size.
   ask_attribute=style, target not in returned list
T6 user: I don't have an additional preference for style.
   ask_attribute=use_case, target not in returned list
T7 user: I don't have an additional preference for use_case.
   ask_attribute=other, target not in returned list
T8 user: For that, what matters is: 100% Polyester.
   ask_attribute=None, target not in returned list
T9 user: Those options are not quite right yet. Ask me about one specific attribute.
   ask_attribute=None, target not in returned list
T10 user: Those options are not quite right yet. Ask me about one specific attribute.
   ask_attribute=None, target not in returned list
```

Returned list on stopping turn:

```text
1. B079JGHJN3 | Amazon Brand - Arabella Women's Shaggy Plush Short Robe | Clothing, Shoes & Jewelry > Women > Clothing > Lingerie, Sleep & Lounge > Sle...
2. B00KHYHRIC | VEAMI Women's Kimono Robe, Short | Clothing, Shoes & Jewelry > Women > Clothing > Lingerie, Sleep & Lounge > Sle...
3. B079JJXSSN | Amazon Brand - Arabella Women's Printed Plush Robe | Clothing, Shoes & Jewelry > Women > Clothing > Lingerie, Sleep & Lounge > Sle...
4. B00F3DOFSW | VEAMI Women's Kimono Robe, Peacock Design, Short | Clothing, Shoes & Jewelry > Women > Clothing > Lingerie, Sleep & Lounge > Sle...
5. B072C3WJT8 | Swhiteme Bridesmaid Robe with Rhinestones, 3/4 Sleeves | Clothing, Shoes & Jewelry > Women > Clothing > Lingerie, Sleep & Lounge > Sle...
6. B00F4O2QC2 | Dearfoams Women's faux sherpa Shawl Collar Long Solid Robe | Clothing, Shoes & Jewelry > Women > Clothing > Lingerie, Sleep & Lounge > Sle...
7. B07DNNZJXN | INTIMO Mens Solid Cozy Plush Robe with Satin Trim | Clothing, Shoes & Jewelry > Men > Clothing > Sleep & Lounge > Robes
8. B01HVLWJ7M | St. Eve Women's Plush Robe | Clothing, Shoes & Jewelry > Women > Clothing > Lingerie, Sleep & Lounge > Sle...
9. B00O3BRNXY | Captain America Costume Hooded Fleece Robe | Clothing, Shoes & Jewelry > Men > Clothing > Sleep & Lounge > Robes
10. B0043RS6B6 | Colorado Clothing Women's Bliss Robe | Clothing, Shoes & Jewelry > Women > Clothing > Lingerie, Sleep & Lounge > Sle...
```

## 4. public_0175 - browsing - miss

Target: `B07D5M61T2 | Ariat Men’s M2 Relaxed Boot Cut Jean | Clothing, Shoes & Jewelry > Men > Clothing > Jeans`

Intent constraints:

```text
hard: cotton; 100% Cotton
soft: Imported; Zipper closure
```

Evaluator result: hit=False, first_hit_turn=None, rank=None, reciprocal_rank=0.000000

Likely problem: Recall miss: even after available clarifications, the target never reached the scored top 10. Generic metadata or late open-ended clarification is part of the pattern.

Conversation:

```text
T1 user: I'm looking for Men Jeans, but I'm still exploring.
   ask_attribute=material, target not in returned list
T2 user: For that, what matters is: cotton; 100% Cotton.
   ask_attribute=color, target not in returned list
T3 user: I don't have an additional preference for color.
   ask_attribute=feature, target not in returned list
T4 user: For that, what matters is: Imported; Zipper closure.
   ask_attribute=brand, target not in returned list
T5 user: I don't have an additional preference for brand.
   ask_attribute=use_case, target not in returned list
T6 user: I don't have an additional preference for use_case.
   ask_attribute=size, target not in returned list
T7 user: I don't have an additional preference for size.
   ask_attribute=style, target not in returned list
T8 user: I don't have an additional preference for style.
   ask_attribute=other, target not in returned list
T9 user: I don't have an additional preference for other.
   ask_attribute=budget, target not in returned list
T10 user: I don't have an additional preference for budget.
   ask_attribute=None, target not in returned list
```

Returned list on stopping turn:

```text
1. B0BZQ82WF9 | Fashion2Love Women's Juniors/Plus Size Middle - Length Stretch Denim Distressed Pencil Skirt... | Clothing, Shoes & Jewelry > Women > Clothing > Skirts > Casual
2. B0B6LK1WBM | Levi's Men's 505 Regular Fit Jeans | Clothing, Shoes & Jewelry > Men > Clothing > Jeans
3. B07BTCGM69 | Hanayome Men's Slim Fit Suit Blazer Jacket Tux Vest & Trousers 3-Piece Suit Set | Clothing, Shoes & Jewelry > Men > Clothing > Suits & Sport Coats > Suits
4. B0BGR4YMDG | CTU Mens Skinny Stretch Ripped Jeans Slim Fit Destroyed Demin Pants with Side Striped Leg Zip | Clothing, Shoes & Jewelry > Men > Clothing > Jeans
5. B01LMZ9GCO | Wrangler Men's Rugged Wear Relaxed Fit Jean | Clothing, Shoes & Jewelry > Men > Clothing > Jeans
6. B08KTSGCTT | Signature by Levi Strauss & Co. Gold Label Men's Straight Fit Jeans, Photon Gold, 36Wx29L | Clothing, Shoes & Jewelry > Men > Clothing > Jeans
7. B07TM8B1XV | Tommy Hilfiger Men's Skinny Fit Jeans | Clothing, Shoes & Jewelry > Men > Clothing > Jeans
8. B09DPY247N | Genleck Womens Baggy Jeans Casual Mom Jeans High Waisted Boyfriend 90s Loose Jeans for Women | Clothing, Shoes & Jewelry > Women > Clothing > Jeans
9. B0754119MX | Rich Cotton Tech Fleece Hoodie Jacket Active Running Hooded Zip Up Workout Hoodie Long Sleev... | Clothing, Shoes & Jewelry > Men > Clothing > Active > Active Hoodies
10. B00DP9L1VA | Sakroots Artist Circle Top Zip Shoulder Bag | Clothing, Shoes & Jewelry > Women > Handbags & Wallets > Shoulder Bags
```

## 5. public_0198 - intent_override - hit turn 9, rank 7

Target: `B08K1ZJZ4N | lola mae Quilted Crossbody Bag, Medium Lightweight Shoulder Purse Top Zipper Tassel Accent | Clothing, Shoes & Jewelry > Women > Handbags & Wallets > Crossbody Bags`

Intent constraints:

```text
hard: leather; color: black
soft: PU; Imported
```

Evaluator result: hit=True, first_hit_turn=9, rank=7, reciprocal_rank=0.142857

Likely problem: Late hit: the agent needed many turns before the target entered the scored list. Generic metadata or late open-ended clarification is part of the pattern.

Conversation:

```text
T1 user: I'm looking for Handbags & Wallets Crossbody Bags. Imported
   ask_attribute=material, target not in returned list
T2 user: For that, what matters is: leather.
   ask_attribute=feature, target not in returned list
T3 user: For that, what matters is: PU; Imported.
   ask_attribute=color, target rank 8
T4 user: Actually, ignore my earlier preference. What I need is: leather.
   ask_attribute=brand, target not in returned list
T5 user: I don't have an additional preference for brand.
   ask_attribute=size, target not in returned list
T6 user: I don't have an additional preference for size.
   ask_attribute=use_case, target not in returned list
T7 user: I don't have an additional preference for use_case.
   ask_attribute=style, target not in returned list
T8 user: I don't have an additional preference for style.
   ask_attribute=other, target not in returned list
T9 user: For that, what matters is: color: black.
   ask_attribute=budget, target rank 7
```

Returned list on stopping turn:

```text
1. B01FCNB9T0 | TrendyFashionJewelry Men Biker Boot Pair Bracelets Black Faux Leather Strap Chain Native Buf... | Clothing, Shoes & Jewelry > Shoe, Jewelry & Watch Accessories > Shoe Care & A...
2. B0BZRCYNCG | JJAI Crossbody Bag Purses for Women Triple Zip Guitar Strap Shoulder Crossbody Handbags With... | Clothing, Shoes & Jewelry > Women > Handbags & Wallets > Crossbody Bags
3. B0BB9NCZ97 | Genda 2Archer Women Leather Tassel Fashion Multipurpose Chest Crossbody Backpack | Clothing, Shoes & Jewelry > Luggage & Travel Gear > Backpacks > Casual Daypacks
4. B071ZMFGTG | Andrew Family Black Monogram Jewelry Gift Boxes Case with LED Light for Ring Earring Pendant... | Clothing, Shoes & Jewelry > Shoe, Jewelry & Watch Accessories > Jewelry Acces...
5. B08ZXTQC8P | Over Earth Genuine Leather Purses and Handbags for Women Crossbody Messenger Purse with Mult... | Clothing, Shoes & Jewelry > Women > Handbags & Wallets > Crossbody Bags
6. B09RJ3JG7M | Nautica Womens Nautica Set a Drift Crossbody, Black, One Size US | Clothing, Shoes & Jewelry > Women > Handbags & Wallets > Crossbody Bags
7. B08K1ZJZ4N | lola mae Quilted Crossbody Bag, Medium Lightweight Shoulder Purse Top Zipper Tassel Accent | Clothing, Shoes & Jewelry > Women > Handbags & Wallets > Crossbody Bags <-- target
8. B07791KH2D | Angelina's Palace Tote bag Shoulder Bag Organizer Insert Waterproof PU Leather Sturdy Zipper... | Clothing, Shoes & Jewelry > Women > Handbags & Wallets > Shoulder Bags
9. B000AI5EBW | Osgoode Marley Cashmere Jotter Wallet - Black | Clothing, Shoes & Jewelry > Women > Accessories > Wallets, Card Cases & Money...
10. B01DUGB1RG | HelloRARA Fashion Korean Style Men's Lace Up Flats High-top Casual Shoes Ankle Boots | Clothing, Shoes & Jewelry > Men > Shoes > Fashion Sneakers
```

## 6. public_0161 - buying - hit turn 9, rank 4

Target: `B0B6N6TJ6V | Thankful Grateful Blessed Shirt for Women Plaid Pumpkin Long Sleeve Tshirts Leopard Stripe S... | Clothing, Shoes & Jewelry > Women > Clothing > Tops, Tees & Blouses > T-Shirts`

Intent constraints:

```text
hard: cotton; cotton blend
soft: Imported; Pull On closure
```

Evaluator result: hit=True, first_hit_turn=9, rank=4, reciprocal_rank=0.250000

Likely problem: Late hit: the agent needed many turns before the target entered the scored list. Generic metadata or late open-ended clarification is part of the pattern.

Conversation:

```text
T1 user: I'm looking for Tees & Blouses T-Shirts. A key requirement is: cotton.
   ask_attribute=color, target not in returned list
T2 user: I don't have an additional preference for color.
   ask_attribute=feature, target not in returned list
T3 user: For that, what matters is: Imported; Pull On closure.
   ask_attribute=brand, target not in returned list
T4 user: I don't have an additional preference for brand.
   ask_attribute=size, target not in returned list
T5 user: I don't have an additional preference for size.
   ask_attribute=use_case, target not in returned list
T6 user: I don't have an additional preference for use_case.
   ask_attribute=style, target not in returned list
T7 user: I don't have an additional preference for style.
   ask_attribute=budget, target not in returned list
T8 user: I don't have an additional preference for budget.
   ask_attribute=other, target not in returned list
T9 user: For that, what matters is: cotton blend.
   ask_attribute=None, target rank 4
```

Returned list on stopping turn:

```text
1. B0C1YPPMQ1 | Womens Mother's Days T Shirt Round Neck Short Sleeve Graphic Tee Tops Mom of Girls Mama Shir... | Clothing, Shoes & Jewelry > Women > Clothing > Tops, Tees & Blouses > T-Shirts
2. B07VV1RV62 | STYLEIE Women Tale As Old As Time Tee Tops Letter Print Graphic Tees Vintage Tanks Tops | Clothing, Shoes & Jewelry > Women > Clothing > Tops, Tees & Blouses > T-Shirts
3. B0BDMJ4TDN | Turkey Face Shirts Women Fall Family Matching Thanksgiving T-Shirt Funny Turkey Graphic Casu... | Clothing, Shoes & Jewelry > Women > Clothing > Tops, Tees & Blouses > T-Shirts
4. B0B6N6TJ6V | Thankful Grateful Blessed Shirt for Women Plaid Pumpkin Long Sleeve Tshirts Leopard Stripe S... | Clothing, Shoes & Jewelry > Women > Clothing > Tops, Tees & Blouses > T-Shirts <-- target
5. B0BW2Y7JNS | MYHALF Plus Size American Flag Shirt Women 4th of July Shirts Patriotic Cold Shoulder T-Shir... | Clothing, Shoes & Jewelry > Women > Clothing > Tops, Tees & Blouses > T-Shirts
6. B08J9TRGJV | Sunflower Shirts for Women Be Kind Printed Short Sleeve Flower Graphic Tee Tops (Pink 51#,S) | Clothing, Shoes & Jewelry > Women > Clothing > Tops, Tees & Blouses > T-Shirts
7. B09XVCFXY8 | Cowboy Take Me Away Shirts for Women Western Vintage Graphic T-Shirt Casual Rodeo Tee Tops… | Clothing, Shoes & Jewelry > Women > Clothing > Tops, Tees & Blouses > T-Shirts
8. B088NP48MV | Sunflower Graphic Tee Shirt Women Cute Flower Tshirts Short Sleeve Casual Tops(Yellow1,S) | Clothing, Shoes & Jewelry > Women > Clothing > Tops, Tees & Blouses > T-Shirts
9. B09PV5HJMB | Good Mom Say Bad Words T Shirt Mom Shirts Short Sleeve Women Funny Letter Print Tee Casual R... | Clothing, Shoes & Jewelry > Women > Clothing > Tops, Tees & Blouses > T-Shirts
10. B01JDBXJKE | Ahh By Rhonda Shear Women's High Waist Cotton Blend Control Legging | Clothing, Shoes & Jewelry > Women > Clothing > Leggings
```

## 7. public_0126 - browsing - hit turn 6, rank 9

Target: `B09M84R91V | Asvivid Womens Casual Boho Floral Print 3/4 Flare Sleeve Blouses Summer Off The Shoulder Top... | Clothing, Shoes & Jewelry > Women > Clothing > Tops, Tees & Blouses > Blouses...`

Intent constraints:

```text
hard: polyester; 95% Polyester, 5% Spandex
soft: Imported; Pull On closure
```

Evaluator result: hit=True, first_hit_turn=6, rank=9, reciprocal_rank=0.111111

Likely problem: Low-rank hit: the target was present, but near-duplicate products outranked it. Generic metadata or late open-ended clarification is part of the pattern.

Conversation:

```text
T1 user: I'm looking for Tees & Blouses Blouses & Button-Down Shirts, but I'm still exploring.
   ask_attribute=material, target not in returned list
T2 user: For that, what matters is: polyester; 95% Polyester, 5% Spandex.
   ask_attribute=feature, target not in returned list
T3 user: For that, what matters is: Imported; Pull On closure.
   ask_attribute=color, target not in returned list
T4 user: I don't have an additional preference for color.
   ask_attribute=size, target not in returned list
T5 user: I don't have an additional preference for size.
   ask_attribute=use_case, target not in returned list
T6 user: I don't have an additional preference for use_case.
   ask_attribute=brand, target rank 9
```

Returned list on stopping turn:

```text
1. B07TZPFQ9N | Biucly Women's Casual V Neck Tops 3/4 Sleeve Tie Knot Chiffon Blouses Button Down Shirts | Clothing, Shoes & Jewelry > Women > Clothing > Tops, Tees & Blouses > Blouses...
2. B07T5D5F6N | BLENCOT Women's Fashion Floral V Neck Long Sleeve Blouses Casual Flowy Bohemian Shirts Tops ... | Clothing, Shoes & Jewelry > Women > Clothing > Tops, Tees & Blouses > Blouses...
3. B087B4KKX7 | BLENCOT Womens 3/4 Bell Sleeve V Neck Lace Patchwork Blouse Casual Loose Shirt Tops | Clothing, Shoes & Jewelry > Women > Clothing > Tops, Tees & Blouses > Blouses...
4. B09KY6NW2D | KIRUNDO 2023 Fall Women's Long Sleeve V Neck Blouses Casual Loose Button Down Solid Knitted ... | Clothing, Shoes & Jewelry > Women > Clothing > Tops, Tees & Blouses > Blouses...
5. B0B3HS4PBQ | Cestyle Womens Ruffle 3/4 Sleeve Tops Mesh Blouses Flowy Shirts Casual Tunic | Clothing, Shoes & Jewelry > Women > Clothing > Tops, Tees & Blouses > Blouses...
6. B0B18MM3RR | JoyJoy Womens Casual Square Neck Tops Off The Shoulder Puff Ruffle Long Sleeve Smocked Blous... | Clothing, Shoes & Jewelry > Women > Clothing > Tops, Tees & Blouses > Blouses...
7. B0B4RW6DF2 | Zeagoo Women's Long Puff Sleeve Square Neck Top Velvet Tops Slim Fit Crop Cropped Top Blouse... | Clothing, Shoes & Jewelry > Women > Clothing > Tops, Tees & Blouses > Blouses...
8. B09B57MHMM | Hanna Nikole Women's Plus Size Long Lantern Sleeve Blouse Side Slit Tunic Tops Casual Office... | Clothing, Shoes & Jewelry > Women > Clothing > Tops, Tees & Blouses > Blouses...
9. B09M84R91V | Asvivid Womens Casual Boho Floral Print 3/4 Flare Sleeve Blouses Summer Off The Shoulder Top... | Clothing, Shoes & Jewelry > Women > Clothing > Tops, Tees & Blouses > Blouses... <-- target
10. B07Y48MQBV | BLENCOT Women's Lightweight Crewneck Twist Knot Long Sleeve Knit Shirts Casual Loose Blouses... | Clothing, Shoes & Jewelry > Women > Clothing > Tops, Tees & Blouses > Blouses...
```

## 8. public_0035 - boundary - hit turn 5, rank 8

Target: `B0BN6CCHB7 | Skechers Men's Go Max-Athletic Air Mesh Slip on Walking Shoe Sneaker | Clothing, Shoes & Jewelry > Men > Shoes > Athletic > Walking`

Intent constraints:

```text
hard: fabric; 100% Textile
soft: Imported; Rubber sole
```

Evaluator result: hit=True, first_hit_turn=5, rank=8, reciprocal_rank=0.125000

Likely problem: Low-rank hit: the target was present, but near-duplicate products outranked it. Generic metadata or late open-ended clarification is part of the pattern.

Conversation:

```text
T1 user: I'm looking for Athletic Walking, but I'm still exploring.
   ask_attribute=material, target not in returned list
T2 user: I don't have a preference for material; please use your judgment.
   ask_attribute=color, target not in returned list
T3 user: I don't have an additional preference for color.
   ask_attribute=feature, target not in returned list
T4 user: For that, what matters is: 100% Textile; Imported.
   ask_attribute=brand, target not in returned list
T5 user: I don't have an additional preference for brand.
   ask_attribute=size, target rank 8
```

Returned list on stopping turn:

```text
1. B00UZ23FRI | Skechers Performance Women's Go Walk Compose Slip-On Walking Shoe | Clothing, Shoes & Jewelry > Women > Shoes > Athletic > Walking
2. B00HF6K5ZM | Merrell Women's All Out Rave Walking Shoe | Clothing, Shoes & Jewelry > Women > Shoes > Athletic > Walking
3. B00ITKSNFM | Skechers Performance Women's Go Walk 2 Extreme Walking Shoe | Clothing, Shoes & Jewelry > Women > Shoes > Athletic > Walking
4. B00MES8EIQ | Skechers Performance Women's Go Walk 3 Fitknit Shoe | Clothing, Shoes & Jewelry > Women > Shoes > Athletic > Walking
5. B078GTVCS6 | Skechers Women's Go Walk Joy-15633 Sneaker | Clothing, Shoes & Jewelry > Women > Shoes > Athletic > Walking
6. B078GGN9SM | Skechers Women's Go Walk Lite-15461 Sneaker | Clothing, Shoes & Jewelry > Women > Shoes > Athletic > Walking
7. B01I42AKS2 | Propét Men's TravelFit Strap Walking Shoe | Clothing, Shoes & Jewelry > Men > Shoes > Athletic > Walking
8. B0BN6CCHB7 | Skechers Men's Go Max-Athletic Air Mesh Slip on Walking Shoe Sneaker | Clothing, Shoes & Jewelry > Men > Shoes > Athletic > Walking <-- target
9. B09JLJP8VS | Skechers Women's Go Run Mojo-Mania Sneaker | Clothing, Shoes & Jewelry > Women > Shoes > Athletic > Walking
10. B00ITKT4XM | Skechers Performance Women's Go Walk 2 Extreme Walking Shoe | Clothing, Shoes & Jewelry > Women > Shoes > Athletic > Walking
```

## 9. public_0087 - browsing - hit turn 5, rank 8

Target: `B0BT158RRR | Goodthreads Men's Standard-Fit Short-Sleeve Printed Poplin Shirt | Clothing, Shoes & Jewelry > Men > Clothing > Shirts > Casual Button-Down Shirts`

Intent constraints:

```text
hard: cotton; 100% Cotton
soft: Imported; Button closure
```

Evaluator result: hit=True, first_hit_turn=5, rank=8, reciprocal_rank=0.125000

Likely problem: Low-rank hit: the target was present, but near-duplicate products outranked it. Generic metadata or late open-ended clarification is part of the pattern.

Conversation:

```text
T1 user: I'm looking for Shirts Casual Button-Down Shirts, but I'm still exploring.
   ask_attribute=material, target not in returned list
T2 user: For that, what matters is: cotton; 100% Cotton.
   ask_attribute=feature, target not in returned list
T3 user: For that, what matters is: Imported; Button closure.
   ask_attribute=color, target not in returned list
T4 user: I don't have an additional preference for color.
   ask_attribute=brand, target not in returned list
T5 user: I don't have an additional preference for brand.
   ask_attribute=size, target rank 8
```

Returned list on stopping turn:

```text
1. B07QW6Z8S4 | Dockers Men's Long Sleeve Alpha Icon Button Down Shirt | Clothing, Shoes & Jewelry > Men > Clothing > Shirts > Casual Button-Down Shirts
2. B06VXNJ4YH | Buttoned Down Men's Slim Fit Button-Collar Supima Cotton Dress Casual Shirt | Clothing, Shoes & Jewelry > Men > Clothing > Shirts > Casual Button-Down Shirts
3. B07KQ4PF54 | LOCALMODE Men's Slim Fit Cotton Business Shirt Solid Long Sleeve Button Down Dress Shirts | Clothing, Shoes & Jewelry > Men > Clothing > Shirts > Casual Button-Down Shirts
4. B083ZPTRRF | Paul Fredrick Men's Slim Fit Cotton Gingham Casual Shirt | Clothing, Shoes & Jewelry > Men > Clothing > Shirts > Casual Button-Down Shirts
5. B085T9Q34Z | TUNEVUSE Mens Flower Shirt Short Sleeve Casual Floral Print Button Down Hawaiian Shirt 100% ... | Clothing, Shoes & Jewelry > Men > Clothing > Shirts > Casual Button-Down Shirts
6. B07NZ8NM7T | Amazon Essentials Men's Short-Sleeve Pocket Oxford Shirt fit by DXL | Clothing, Shoes & Jewelry > Men > Clothing > Shirts > Casual Button-Down Shirts
7. B07SHVBHXL | Cinch Men's Classic Fit Long Sleeve Button One Open Pocket Shirt | Clothing, Shoes & Jewelry > Men > Clothing > Shirts > Casual Button-Down Shirts
8. B0BT158RRR | Goodthreads Men's Standard-Fit Short-Sleeve Printed Poplin Shirt | Clothing, Shoes & Jewelry > Men > Clothing > Shirts > Casual Button-Down Shirts <-- target
9. B09QB2NP99 | DXL Big & Tall Essentials Men's Plaid Poplin Shirt | 100% Cotton Short-Sleeve Button Down Sh... | Clothing, Shoes & Jewelry > Men > Clothing > Shirts > Casual Button-Down Shirts
10. B07KQ36LVT | Buffalo David Bitton Men's Long Sleeve Button Down Light Plaid Shirt | Clothing, Shoes & Jewelry > Men > Clothing > Shirts > Casual Button-Down Shirts
```

## 10. public_0137 - browsing - hit turn 6, rank 5

Target: `B01N67CJGX | DOUBLJU Lightweight Thin Zip-Up Hoodie Jacket for Women Girls Kids with Plus Size | Clothing, Shoes & Jewelry > Women > Clothing > Fashion Hoodies & Sweatshirts`

Intent constraints:

```text
hard: cotton; 60% Cotton, 40% Polyester
soft: Zipper closure; Machine Wash
```

Evaluator result: hit=True, first_hit_turn=6, rank=5, reciprocal_rank=0.200000

Likely problem: Moderate issue: target was found, but still left scoring headroom. Generic metadata or late open-ended clarification is part of the pattern.

Conversation:

```text
T1 user: I'm looking for Women Fashion Hoodies & Sweatshirts, but I'm still exploring.
   ask_attribute=material, target not in returned list
T2 user: For that, what matters is: cotton; 60% Cotton, 40% Polyester.
   ask_attribute=color, target not in returned list
T3 user: I don't have an additional preference for color.
   ask_attribute=feature, target not in returned list
T4 user: For that, what matters is: Zipper closure; Machine Wash.
   ask_attribute=brand, target not in returned list
T5 user: I don't have an additional preference for brand.
   ask_attribute=use_case, target not in returned list
T6 user: I don't have an additional preference for use_case.
   ask_attribute=style, target rank 5
```

Returned list on stopping turn:

```text
1. B08VR8ZKFV | ARMADIO Women's zip up hoodie l workout hoodie for women | Clothing, Shoes & Jewelry > Novelty & More > Clothing > Novelty > Women > Hoo...
2. B01FV283GE | Element Juniors Duel Pullover Crew Burn Out Fleece | Clothing, Shoes & Jewelry > Women > Clothing > Fashion Hoodies & Sweatshirts
3. B09FLDVFPW | LASLULU Mens Zip Up Hoodies Long Sleeve Casual Sweatshirt Drawstring Fleece Hooded Pullover ... | Clothing, Shoes & Jewelry > Men > Clothing > Fashion Hoodies & Sweatshirts
4. B08LSFCGBQ | Women's Warm Fashion wear Fleece Sweat suit Sweatpants and Jacket Set | Clothing, Shoes & Jewelry > Women > Clothing > Active > Sets > Active Sweatsuits
5. B01N67CJGX | DOUBLJU Lightweight Thin Zip-Up Hoodie Jacket for Women Girls Kids with Plus Size | Clothing, Shoes & Jewelry > Women > Clothing > Fashion Hoodies & Sweatshirts <-- target
6. B095QW81G7 | Champion Boys Fleece Hooded Zip Up Sweatshirt Hoody Kids Clothes | Clothing, Shoes & Jewelry > Boys > Clothing > Active > Active Hoodies
7. B07QGTHJBV | JoJo Siwa Girls' Little Unicorns & Rainbows All Over Print Zip Up Hoodie with Bow | Clothing, Shoes & Jewelry > Women > Clothing > Fashion Hoodies & Sweatshirts
8. B01N18JPH5 | Baby Boys Long Sleeve Dinosaur Hoodies Kids Sweatshirt Toddler Zip-up Jacket | Clothing, Shoes & Jewelry > Baby > Baby Boys > Clothing > Hoodies & Active
9. B00H4MRO8S | PIZOFF Unisex Luxury Baroque Golden Medusa Bomber Silk Blend Smooth Light Jacket MA-1 | Clothing, Shoes & Jewelry > Men > Clothing > Jackets & Coats > Lightweight Ja...
10. B00KMS0ST8 | Champion Men's Style Jogger Pant | Clothing, Shoes & Jewelry > Men > Clothing > Active > Active Pants
```

## Transferable patterns

1. The remaining loss is mostly rank/MRR, not raw hit rate. Hit@10 is already 0.980.
2. Buying sessions have the weakest MRR. Many buying misses are crowded commodity categories where the first disclosed constraint is common.
3. Intent override still has high MTTC because conversion cannot happen before the override, and reset logic can lose useful category context.
4. Generic metadata such as `Imported`, `Machine Wash`, `Hand Wash Only`, `Pull On closure`, and model/date fields is noisy; it helps only when attached to strong category/title anchoring.
5. The strongest transferable improvements are better near-duplicate reranking, safer category anchoring after override, and earlier `other` only when specific attributes have low expected value.
