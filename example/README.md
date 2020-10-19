# Example

## Example from CNN/DM dataset

### Article

```
(CNN) For the first time in eight years, a TV legend returned to doing what he does best. Contestants told to "come on down!" on the April 1 edition of "The Price Is Right" encountered not host Drew Carey but another familiar face in charge of the proceedings. Instead, there was Bob Barker, who hosted the TV game show for 35 years before stepping down in 2007. Looking spry at 91, Barker handled the first price - guessing game of the show, the classic " Lucky Seven," before turning hosting duties over to Carey, who finished up. Despite being away from the show for most of the past eight years, Barker didn' t seem to miss a beat.
```

### Ground Truth

```
Bob Barker returned to host "The Price Is Right" on Wednesday. Barker, 91, had retired as host in 2007.
```

## Summary

|   Model    |                           Summary                            |
| :--------: | :----------------------------------------------------------: |
|    BART    | Bob Barker hosted the TV game show for 35 years before stepping down in 2007. Barker handled the first price-guessing game of the show before turning hosting duties over to Drew Carey. Despite being away from the show for most of the past eight years, Barker didn't seem to miss a beat. |
|  PEGASUS   | Barker hosted "The Price Is Right" for 35 years. He stepped down in 2007. |
| ProphetNet | A TV legend returned to doing what he does best. Contestants told to "come on down!" on the April 1 edition of "The Price Is Right" encountered not host Drew Xarey. |
|     T5     | "The Price Is Right" returned to hosting for the first time in eight years. Despite being away from the show for most of the past 8 years, a television legend didn't seem to miss a beat.  Bob Barker hosted the game show for 35 years before stepping down in 2007. |
|  **TAAS**  | Bob Barker returns to hosting "The Price Is Right" for the first time in eight years. The 91-year-old TV legend stepped down from the show in 2007 after 35 years on the show, which he hosted for 35 years. |





