class Solution {
public:
    long long countSubarrays(vector<long long>& arr, long long k) {
        long long maxi = 0,ans = 0,l = 0;
        for(auto it:arr) maxi = max(maxi,it);
        vector<long long> loc;
        long long cnt = 0;
        for(long long i=0;i<arr.size();i++){ 
            if(arr[i] == maxi) cnt++;
            loc.push_back(cnt);
        }
        
        long long q = 0,p=loc.size();
        for(long long i=0;i<loc.size();i++){
            if(loc[i] == k){
                p = i;
                break;
            }
        }
        // cout<<p<<q;
        while(p!=arr.size()){
            while(loc[p] - loc[q] >= k) q++;
            ans += q+1;
            p++;
        }
        return ans;
    }
};